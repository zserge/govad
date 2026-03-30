package govad

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

const refPath = "testdata/reference.bin"

func TestReflectPadRight(t *testing.T) {
	in := []float32{0, 1, 2, 3, 4}
	got := reflectPadRight(in, 3)
	want := []float32{0, 1, 2, 3, 4, 3, 2, 1}
	if len(got) != len(want) {
		t.Fatalf("len: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("index %d: got %f, want %f", i, got[i], want[i])
		}
	}
}

func TestLoadWeights(t *testing.T) {
	v, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	checks := []struct {
		name string
		got  int
		want int
	}{
		{"stftW", len(v.stftW), 258 * 256},
		{"conv1W", len(v.conv1W), 128 * 129 * 3},
		{"conv1B", len(v.conv1B), 128},
		{"conv2W", len(v.conv2W), 64 * 128 * 3},
		{"conv2B", len(v.conv2B), 64},
		{"conv3W", len(v.conv3W), 64 * 64 * 3},
		{"conv3B", len(v.conv3B), 64},
		{"conv4W", len(v.conv4W), 128 * 64 * 3},
		{"conv4B", len(v.conv4B), 128},
		{"lstmWIH", len(v.lstmWIH), 512 * 128},
		{"lstmWHH", len(v.lstmWHH), 512 * 128},
		{"lstmBIH", len(v.lstmBIH), 512},
		{"lstmBHH", len(v.lstmBHH), 512},
		{"finalW", len(v.finalW), 128},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s: got %d, want %d", c.name, c.got, c.want)
		}
	}
}

// TestAgainstONNX runs the Go model on the same audio chunks used in the
// Python test and compares against ONNX reference probabilities.
func TestAgainstONNX(t *testing.T) {
	v, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	f, err := os.Open(refPath)
	if err != nil {
		t.Fatalf("open reference: %v", err)
	}
	defer f.Close()

	var nFrames uint32
	if err := binary.Read(f, binary.LittleEndian, &nFrames); err != nil {
		t.Fatalf("read nFrames: %v", err)
	}

	chunks := make([][]float32, nFrames)
	for i := uint32(0); i < nFrames; i++ {
		chunk, err := readF32(f, SamplesPerFrame)
		if err != nil {
			t.Fatalf("read chunk %d: %v", i, err)
		}
		chunks[i] = chunk
	}

	expectedProbs, err := readF32(f, int(nFrames))
	if err != nil {
		t.Fatalf("read probs: %v", err)
	}

	var maxDiff float64
	for i := uint32(0); i < nFrames; i++ {
		got := v.Process(chunks[i])
		want := expectedProbs[i]
		diff := math.Abs(float64(got) - float64(want))
		if diff > maxDiff {
			maxDiff = diff
		}
		t.Logf("Frame %2d: got=%.6f  want=%.6f  diff=%.6f", i, got, want, diff)
		if diff > 0.001 {
			t.Errorf("Frame %d: got %.6f, want %.6f (diff %.6f > 0.001)", i, got, want, diff)
		}
	}
	t.Logf("Max absolute diff: %.6f", maxDiff)
}

func BenchmarkProcess(b *testing.B) {
	v, err := New()
	if err != nil {
		b.Fatalf("New: %v", err)
	}

	samples := make([]float32, SamplesPerFrame)
	for i := range samples {
		samples[i] = float32(math.Sin(float64(i) * 2 * math.Pi * 440 / 16000))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v.Process(samples)
	}
}
