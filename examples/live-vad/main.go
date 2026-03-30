// Command live-vad captures audio from the default microphone at 16 kHz and
// prints real-time voice activity detection results using the Silero VAD model.
//
// Usage:
//
//	go run . [-weights path/to/weights.bin] [-threshold 0.5]
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"os/signal"
	"strings"
	"unsafe"

	"github.com/gen2brain/malgo"

	"github.com/zserge/govad"
)

func main() {
	weightsPath := flag.String("weights", "", "path to custom weights.bin (uses embedded model if empty)")
	threshold := flag.Float64("threshold", 0.5, "speech probability threshold")
	flag.Parse()

	var v *govad.VAD
	var err error
	if *weightsPath != "" {
		v, err = govad.NewFromFile(*weightsPath)
	} else {
		v, err = govad.New()
	}
	if err != nil {
		log.Fatalf("load model: %v", err)
	}

	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, nil)
	if err != nil {
		log.Fatalf("init audio context: %v", err)
	}
	defer ctx.Uninit()
	defer ctx.Free()

	deviceConfig := malgo.DefaultDeviceConfig(malgo.Capture)
	deviceConfig.Capture.Format = malgo.FormatF32
	deviceConfig.Capture.Channels = 1
	deviceConfig.SampleRate = 16000

	var buf [govad.SamplesPerFrame]float32
	var pos int
	var wasSpeech bool

	onData := func(_, input []byte, frameCount uint32) {
		samples := unsafe.Slice((*float32)(unsafe.Pointer(&input[0])), frameCount)

		for _, s := range samples {
			buf[pos] = s
			pos++
			if pos == govad.SamplesPerFrame {
				prob := v.Process(buf[:])
				isSpeech := float64(prob) >= *threshold

				if isSpeech != wasSpeech {
					bar := renderBar(prob)
					if isSpeech {
						fmt.Printf("\r\033[K\033[32m● SPEECH\033[0m  p=%.3f %s\n", prob, bar)
					} else {
						fmt.Printf("\r\033[K\033[31m○ SILENT\033[0m  p=%.3f %s\n", prob, bar)
					}
					wasSpeech = isSpeech
				} else if isSpeech {
					bar := renderBar(prob)
					fmt.Printf("\r\033[K\033[32m● SPEECH\033[0m  p=%.3f %s", prob, bar)
				}

				pos = 0
			}
		}
	}

	callbacks := malgo.DeviceCallbacks{Data: onData}
	device, err := malgo.InitDevice(ctx.Context, deviceConfig, callbacks)
	if err != nil {
		log.Fatalf("init capture device: %v", err)
	}
	defer device.Uninit()

	if err := device.Start(); err != nil {
		log.Fatalf("start capture: %v", err)
	}

	fmt.Println("Listening on default microphone (16 kHz). Press Ctrl-C to stop.")
	fmt.Printf("Threshold: %.2f\n\n", *threshold)

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt)
	<-sig
	fmt.Println("\nStopped.")
}

func renderBar(prob float32) string {
	const width = 30
	n := int(math.Round(float64(prob) * width))
	if n < 0 {
		n = 0
	}
	if n > width {
		n = width
	}
	return "[" + strings.Repeat("█", n) + strings.Repeat("░", width-n) + "]"
}
