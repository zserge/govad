# govad

Pure Go voice activity detection using the [Silero VAD](https://github.com/snakers4/silero-vad) neural network.

No CGo. No ONNX runtime. No external dependencies.

The model weights are embedded in the binary.

## Features

- Pure Go inference (~300 lines), zero C dependencies
- Processes 512-sample frames (32 ms at 16 kHz)
- Stateful LSTM — feed frames sequentially, get speech probabilities
- Embedded model weights — no extra files to ship
- Validated against the ONNX reference (max diff < 0.001)

## Installation

```
go get github.com/zserge/govad@latest
```

## Usage

```go
package main

import (
	"fmt"
	"github.com/zserge/govad"
)

func main() {
	// Create a VAD detector (uses embedded weights)
	v, err := govad.New()
	if err != nil {
		panic(err)
	}

	// Feed 512 float32 samples at 16 kHz per call
	samples := make([]float32, govad.SamplesPerFrame)
	// ... fill samples from your audio source ...

	prob := v.Process(samples)
	if prob > 0.5 {
		fmt.Println("Speech detected!")
	}

	// Call Reset() between unrelated audio streams
	v.Reset()
}
```

## Live microphone example

The `examples/live-vad` directory contains a complete real-time VAD demo
using [malgo](https://github.com/gen2brain/malgo) (miniaudio bindings):

```
cd examples/live-vad
go run . -threshold 0.5
```

It captures audio from your default microphone and prints speech/silence
transitions in real time.

## API

| Function | Description |
|----------|-------------|
| `govad.New()` | Create a detector with embedded default weights |
| `govad.NewFromFile(path)` | Load weights from a file |
| `govad.NewFromReader(r)` | Load weights from an `io.Reader` |
| `v.Process(samples)` | Run inference on 512 samples, returns probability `[0, 1]` |
| `v.Reset()` | Clear LSTM state for a new audio stream |

## Performance

On Apple M1:

```
BenchmarkProcess-8    1911    632370 ns/op    10112 B/op    7 allocs/op
```

~632 µs per 32 ms frame — roughly 50× faster than real time.

## Model

The weights are exported from `silero_vad_half.onnx` (Silero VAD v5, 16 kHz only).
The architecture is:

```
Audio (512 samples, 16 kHz)
  → Reflect pad (64 right)
  → Conv-STFT (n_fft=256, hop=128)
  → Magnitude spectrum
  → Conv1d(129→128, k=3) + ReLU
  → Conv1d(128→64,  k=3, stride=2) + ReLU
  → Conv1d(64→64,   k=3, stride=2) + ReLU
  → Conv1d(64→128,  k=3) + ReLU
  → LSTMCell(128)
  → ReLU → Linear(128→1) → Sigmoid
  → Speech probability
```

## License

The Go code is MIT licensed. The model weights are from [Silero VAD](https://github.com/snakers4/silero-vad), also MIT licensed.
