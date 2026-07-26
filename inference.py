#!/usr/bin/env python3
"""
Loom Inference CLI
Generate video from text prompt using the Loom corpus.
Pure CPU, no GPU required. No neural networks.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from loom import LoomEngine


def save_video(frames: np.ndarray, output_path: str, fps: float = 24.0):
    """Save frames as MP4 video."""
    t, h, w, c = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)
    
    writer.release()
    print(f"Saved {t} frames to {output_path}")


def save_frames(frames: np.ndarray, output_dir: str):
    """Save individual frames as PNG images."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for i, frame in enumerate(frames):
        cv2.imwrite(str(out_path / f"frame_{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    print(f"Saved {len(frames)} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Loom -- Non-parametric video synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --vid corpus.vid --prompt "ocean waves at sunset" --output waves.mp4
  python inference.py --vid corpus.vid --prompt "forest path with dolly camera" --width 1280 --height 720
  python inference.py --vid corpus.vid --prompt "neon city night" --frames 240 --fps 30
        """
    )
    
    parser.add_argument("--vid", required=True, help="Path to .vid corpus file")
    parser.add_argument("--prompt", required=True, help="Text prompt describing the video")
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument("--width", type=int, default=854, help="Video width in pixels")
    parser.add_argument("--height", type=int, default=480, help="Video height in pixels")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames to generate")
    parser.add_argument("--fps", type=float, default=24.0, help="Output frame rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--save-frames", action="store_true", help="Also save individual frames")
    parser.add_argument("--frames-dir", default="frames", help="Directory for individual frames")
    
    args = parser.parse_args()
    
    if not Path(args.vid).exists():
        print(f"Error: Corpus file not found: {args.vid}")
        sys.exit(1)
    
    if args.width < 64 or args.height < 64:
        print("Error: Resolution too small. Minimum 64x64.")
        sys.exit(1)
    
    if args.frames < 1:
        print("Error: Must generate at least 1 frame.")
        sys.exit(1)
    
    print("=" * 60)
    print("Loom Video Synthesizer")
    print("=" * 60)
    print(f"Corpus:    {args.vid}")
    print(f"Prompt:    \"{args.prompt}\"")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Duration:  {args.frames} frames @ {args.fps}fps = {args.frames/args.fps:.1f}s")
    print(f"Seed:      {args.seed if args.seed is not None else 'random'}")
    print("=" * 60)
    
    print("\nInitializing Loom engine...")
    start_load = time.time()
    engine = LoomEngine(args.vid)
    print(f"   Loaded in {time.time() - start_load:.2f}s")
    
    print("\nSynthesizing video...")
    start_gen = time.time()
    
    try:
        frames = engine.synthesize(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_frames=args.frames,
            seed=args.seed
        )
    except Exception as e:
        print(f"\nError: Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    elapsed = time.time() - start_gen
    fps_generated = args.frames / elapsed
    
    print(f"\nGenerated {len(frames)} frames in {elapsed:.1f}s")
    print(f"   Speed: {fps_generated:.1f} frames/sec")
    print(f"   Memory: {frames.nbytes / (1024*1024):.1f} MB (raw)")
    
    print("\nSaving output...")
    save_video(frames, args.output, fps=args.fps)
    
    if args.save_frames:
        save_frames(frames, args.frames_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
