// Package govad provides voice activity detection (VAD) for 16 kHz mono audio.
//
// It is a pure Go inference implementation of the Silero VAD neural network —
// no C dependencies, no ONNX runtime, no CGO required.
//
// # Quick start
//
// The package embeds default model weights, so getting started is a single call:
//
//	v, _ := govad.New()
//	prob := v.Process(samples512) // returns speech probability [0, 1]
//
// For custom weights exported from a different Silero VAD ONNX model,
// use [NewFromFile] or [NewFromReader].
//
// # Input format
//
// Each call to [VAD.Process] expects exactly 512 float32 samples of
// 16 kHz mono audio (32 ms per frame). Samples should be normalised
// to the range [−1, 1]. The detector maintains internal LSTM state
// across calls; use [VAD.Reset] to start a new audio stream.
//
// # Model architecture
//
// Conv-STFT (n_fft=256) → magnitude → 4× Conv1d+ReLU → LSTMCell(128) → Conv1d(1) → Sigmoid
//
// Weights are derived from silero_vad_half.onnx (Silero VAD v5, 16 kHz,
// MIT-licensed). See https://github.com/snakers4/silero-vad for the
// original model.
package govad

import (
	"bytes"
	_ "embed"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

//go:embed model/silero_vad.bin
var defaultWeights []byte

const (
	// SamplesPerFrame is the number of float32 audio samples per inference frame.
	SamplesPerFrame = 512

	nFFT    = 256
	hopSize = 128        // STFT stride
	padSize = 64         // reflect-pad on the right
	cutoff  = nFFT/2 + 1 // 129 (positive frequencies including DC and Nyquist)
	stftCh  = 2 * cutoff // 258 (real + imaginary)
	hidden  = 128        // LSTM hidden size
)

// VAD performs voice activity detection on 16 kHz mono audio.
//
// A VAD instance is not safe for concurrent use. Create one per goroutine,
// or protect calls with a mutex.
type VAD struct {
	// Model weights (flat, row-major)
	stftW   []float32 // [258 × 256]
	conv1W  []float32 // [128 × 129 × 3]
	conv1B  []float32 // [128]
	conv2W  []float32 // [64 × 128 × 3]
	conv2B  []float32 // [64]
	conv3W  []float32 // [64 × 64 × 3]
	conv3B  []float32 // [64]
	conv4W  []float32 // [128 × 64 × 3]
	conv4B  []float32 // [128]
	lstmWIH []float32 // [512 × 128]
	lstmWHH []float32 // [512 × 128]
	lstmBIH []float32 // [512]
	lstmBHH []float32 // [512]
	finalW  []float32 // [128]
	finalB  float32

	// LSTM hidden and cell state, persisted across calls.
	h [hidden]float32
	c [hidden]float32
}

// New creates a VAD detector using the embedded default model weights.
func New() (*VAD, error) {
	return NewFromReader(bytes.NewReader(defaultWeights))
}

// NewFromFile creates a VAD detector by loading model weights from a file.
func NewFromFile(path string) (*VAD, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("govad: open weights: %w", err)
	}
	defer f.Close()
	return NewFromReader(f)
}

// NewFromReader creates a VAD detector by reading model weights from r.
// The binary format is a sequence of little-endian float32 values in the
// order produced by the export_for_go.py script.
func NewFromReader(r io.Reader) (*VAD, error) {
	v := &VAD{}
	var err error

	if v.stftW, err = readF32(r, 258*1*256); err != nil {
		return nil, fmt.Errorf("stftW: %w", err)
	}
	if v.conv1W, err = readF32(r, 128*129*3); err != nil {
		return nil, fmt.Errorf("conv1W: %w", err)
	}
	if v.conv1B, err = readF32(r, 128); err != nil {
		return nil, fmt.Errorf("conv1B: %w", err)
	}
	if v.conv2W, err = readF32(r, 64*128*3); err != nil {
		return nil, fmt.Errorf("conv2W: %w", err)
	}
	if v.conv2B, err = readF32(r, 64); err != nil {
		return nil, fmt.Errorf("conv2B: %w", err)
	}
	if v.conv3W, err = readF32(r, 64*64*3); err != nil {
		return nil, fmt.Errorf("conv3W: %w", err)
	}
	if v.conv3B, err = readF32(r, 64); err != nil {
		return nil, fmt.Errorf("conv3B: %w", err)
	}
	if v.conv4W, err = readF32(r, 128*64*3); err != nil {
		return nil, fmt.Errorf("conv4W: %w", err)
	}
	if v.conv4B, err = readF32(r, 128); err != nil {
		return nil, fmt.Errorf("conv4B: %w", err)
	}
	if v.lstmWIH, err = readF32(r, 512*128); err != nil {
		return nil, fmt.Errorf("lstmWIH: %w", err)
	}
	if v.lstmWHH, err = readF32(r, 512*128); err != nil {
		return nil, fmt.Errorf("lstmWHH: %w", err)
	}
	if v.lstmBIH, err = readF32(r, 512); err != nil {
		return nil, fmt.Errorf("lstmBIH: %w", err)
	}
	if v.lstmBHH, err = readF32(r, 512); err != nil {
		return nil, fmt.Errorf("lstmBHH: %w", err)
	}
	if v.finalW, err = readF32(r, 128); err != nil {
		return nil, fmt.Errorf("finalW: %w", err)
	}
	fb, err := readF32(r, 1)
	if err != nil {
		return nil, fmt.Errorf("finalB: %w", err)
	}
	v.finalB = fb[0]

	return v, nil
}

// Reset clears the LSTM state so the next [VAD.Process] call starts
// a fresh audio stream. Call this between unrelated audio segments.
func (v *VAD) Reset() {
	v.h = [hidden]float32{}
	v.c = [hidden]float32{}
}

// Process runs inference on exactly [SamplesPerFrame] (512) float32 samples
// of 16 kHz mono audio and returns the speech probability in [0.0, 1.0].
//
// The detector maintains LSTM state across calls, so frames should be
// fed in chronological order. A probability above 0.5 typically indicates
// speech; tune the threshold to your use case.
//
// Process panics if len(samples) != [SamplesPerFrame].
func (v *VAD) Process(samples []float32) float32 {
	if len(samples) != SamplesPerFrame {
		panic(fmt.Sprintf("govad: expected %d samples, got %d", SamplesPerFrame, len(samples)))
	}

	// 1. Reflect-pad 64 samples on the right: 512 → 576
	padded := reflectPadRight(samples, padSize)
	paddedLen := len(padded) // 576

	// 2. STFT convolution: [1 × 576] → [258 × 3]
	stftTime := (paddedLen-nFFT)/hopSize + 1 // 3
	stftOut := conv1d(padded, 1, paddedLen, v.stftW, stftCh, nFFT, nil, hopSize, 0)

	// 3. Magnitude spectrum: sqrt(real² + imag²), [258 × 3] → [129 × 3]
	mag := magnitudeSpectrum(stftOut, cutoff, stftTime)

	// 4. Encoder: 4 Conv1d layers with ReLU
	// conv1: [129 × 3] → [128 × 3], kernel=3, stride=1, pad=1
	c1 := conv1d(mag, cutoff, stftTime, v.conv1W, 128, 3, v.conv1B, 1, 1)
	c1Time := stftTime // 3
	reluInPlace(c1)

	// conv2: [128 × 3] → [64 × 2], kernel=3, stride=2, pad=1
	c2 := conv1d(c1, 128, c1Time, v.conv2W, 64, 3, v.conv2B, 2, 1)
	c2Time := (c1Time+2*1-3)/2 + 1 // 2
	reluInPlace(c2)

	// conv3: [64 × 2] → [64 × 1], kernel=3, stride=2, pad=1
	c3 := conv1d(c2, 64, c2Time, v.conv3W, 64, 3, v.conv3B, 2, 1)
	c3Time := (c2Time+2*1-3)/2 + 1 // 1
	reluInPlace(c3)

	// conv4: [64 × 1] → [128 × 1], kernel=3, stride=1, pad=1
	c4 := conv1d(c3, 64, c3Time, v.conv4W, 128, 3, v.conv4B, 1, 1)
	_ = (c3Time+2*1-3)/1 + 1 // 1
	reluInPlace(c4)

	// 5. LSTM cell: input is the 128-dim feature vector (time dim = 1, squeezed)
	v.lstmCell(c4[:hidden])

	// 6. Output head: ReLU(h) → linear(128→1) → sigmoid
	var logit float32
	for i := 0; i < hidden; i++ {
		hi := v.h[i]
		if hi < 0 {
			hi = 0 // ReLU
		}
		logit += hi * v.finalW[i]
	}
	logit += v.finalB

	return sigmoidf(logit)
}

// --- internal helpers ---

func reflectPadRight(x []float32, pad int) []float32 {
	n := len(x)
	out := make([]float32, n+pad)
	copy(out, x)
	for i := 0; i < pad; i++ {
		out[n+i] = x[n-2-i]
	}
	return out
}

// conv1d performs 1-D convolution on flat-packed data.
//
//	input:  [inCh × inLen]     row-major
//	weight: [outCh × inCh × k] row-major
//	bias:   [outCh] or nil
//
// Returns [outCh × outLen] flat, where outLen = (inLen + 2·padding − k) / stride + 1.
func conv1d(input []float32, inCh, inLen int, weight []float32, outCh, k int, bias []float32, stride, padding int) []float32 {
	outLen := (inLen+2*padding-k)/stride + 1
	out := make([]float32, outCh*outLen)

	for oc := 0; oc < outCh; oc++ {
		var b float32
		if bias != nil {
			b = bias[oc]
		}
		for t := 0; t < outLen; t++ {
			sum := b
			for ic := 0; ic < inCh; ic++ {
				wBase := oc*inCh*k + ic*k
				iBase := ic * inLen
				startPos := t*stride - padding
				for ki := 0; ki < k; ki++ {
					pos := startPos + ki
					if pos >= 0 && pos < inLen {
						sum += weight[wBase+ki] * input[iBase+pos]
					}
				}
			}
			out[oc*outLen+t] = sum
		}
	}
	return out
}

// magnitudeSpectrum computes sqrt(real² + imag²) from an STFT output.
// stft has layout [stftCh × timeLen]; first cutoff channels are real, next are imaginary.
func magnitudeSpectrum(stft []float32, cutoff, timeLen int) []float32 {
	out := make([]float32, cutoff*timeLen)
	for i := 0; i < cutoff; i++ {
		for t := 0; t < timeLen; t++ {
			r := stft[i*timeLen+t]
			im := stft[(cutoff+i)*timeLen+t]
			out[i*timeLen+t] = float32(math.Sqrt(float64(r*r + im*im)))
		}
	}
	return out
}

func reluInPlace(x []float32) {
	for i := range x {
		if x[i] < 0 {
			x[i] = 0
		}
	}
}

// lstmCell updates v.h and v.c in place.
// x is the 128-dim input vector.
func (v *VAD) lstmCell(x []float32) {
	var gates [4 * hidden]float32

	for j := 0; j < 4*hidden; j++ {
		sum := v.lstmBIH[j] + v.lstmBHH[j]
		base := j * hidden
		for k := 0; k < hidden; k++ {
			sum += x[k]*v.lstmWIH[base+k] + v.h[k]*v.lstmWHH[base+k]
		}
		gates[j] = sum
	}

	for i := 0; i < hidden; i++ {
		ig := sigmoidf(gates[i])          // input gate
		fg := sigmoidf(gates[hidden+i])   // forget gate
		gg := tanhf(gates[2*hidden+i])    // cell gate
		og := sigmoidf(gates[3*hidden+i]) // output gate
		v.c[i] = fg*v.c[i] + ig*gg
		v.h[i] = og * tanhf(v.c[i])
	}
}

func sigmoidf(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

func tanhf(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

func readF32(r io.Reader, n int) ([]float32, error) {
	buf := make([]byte, n*4)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
	}
	return out, nil
}
