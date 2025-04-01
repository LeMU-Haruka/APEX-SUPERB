import soundfile as sf
import numpy as np
import argparse
import os
import glob
import random
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress specific UserWarning from soundfile about metadata chunks if they occur
warnings.filterwarnings("ignore", message="Chunk format requires 'requires_fseek' True for data read", category=UserWarning)

def calculate_rms(audio):
    """Calculates the Root Mean Square of the audio signal."""
    return np.sqrt(np.mean(audio**2))

def calculate_scaling_factor(speech_rms, noise_rms, target_snr_db, epsilon=1e-10):
    """Calculates the scaling factor for the noise based on target SNR."""
    # SNR = 20 * log10(RMS_speech / RMS_noise)
    # target_snr_db = 20 * log10(speech_rms / (scaling_factor * noise_rms))
    # 10**(target_snr_db / 20) = speech_rms / (scaling_factor * noise_rms)
    # scaling_factor = speech_rms / (noise_rms * 10**(target_snr_db / 20))

    if speech_rms < epsilon or noise_rms < epsilon:
        # Avoid division by zero or amplifying silence unreasonably
        # If either signal is essentially silent, return a very small factor
        # Or handle as an error/skip depending on requirements.
        # Here, we prevent amplification of noise if speech is silent.
        return 0.0 if speech_rms < epsilon else 1.0 # Or adjust logic as needed

    snr_linear = 10**(target_snr_db / 10) # SNR in power ratio
    # Alternatively, using RMS: snr_linear_amplitude = 10**(target_snr_db / 20)
    # scaling_factor = speech_rms / (noise_rms * snr_linear_amplitude)

    # Using power ratio definition: SNR = Power_speech / Power_noise
    # Power = RMS^2
    # target_snr_power = speech_rms**2 / (scaling_factor**2 * noise_rms**2)
    # scaling_factor**2 = speech_rms**2 / (noise_rms**2 * target_snr_power)
    # scaling_factor = speech_rms / (noise_rms * sqrt(target_snr_power))
    scaling_factor = speech_rms / (noise_rms * np.sqrt(snr_linear))

    return scaling_factor

def mix_audio(speech_path, noise_path, target_snr_db, output_dir):
    """
    Loads speech and noise, mixes them at the target SNR, and saves the result.
    Returns the path to the saved file or None if an error occurs.
    """
    try:
        # --- Load Audio ---
        speech_info = sf.info(speech_path)
        noise_info = sf.info(noise_path)

        # --- Sample Rate Check ---
        if speech_info.samplerate != noise_info.samplerate:
            print(f"Warning: Sample rate mismatch! Speech '{Path(speech_path).name}' ({speech_info.samplerate} Hz) "
                  f"vs Noise '{Path(noise_path).name}' ({noise_info.samplerate} Hz). Skipping this pair.")
            print("         Please ensure all files have the same sample rate before mixing.")
            return None

        samplerate = speech_info.samplerate
        speech, _ = sf.read(speech_path, dtype='float32')
        noise, _ = sf.read(noise_path, dtype='float32')

        # Ensure mono audio (take first channel if stereo)
        if speech.ndim > 1:
            speech = speech[:, 0]
        if noise.ndim > 1:
            noise = noise[:, 0]

        speech_len = len(speech)
        noise_len = len(noise)

        # --- Length Matching ---
        if speech_len == 0:
             print(f"Warning: Speech file '{Path(speech_path).name}' is empty. Skipping.")
             return None

        if noise_len == 0:
             print(f"Warning: Noise file '{Path(noise_path).name}' is empty. Trying another noise file if possible, else skipping.")
             # In a real loop, you'd likely select a different noise file here.
             # For this single function call, we'll just return None.
             return None

        if noise_len < speech_len:
            # Repeat noise if shorter than speech
            repeats = int(np.ceil(speech_len / noise_len))
            noise = np.tile(noise, repeats)[:speech_len]
        elif noise_len > speech_len:
            # Take a random segment of noise if longer than speech
            start = random.randint(0, noise_len - speech_len)
            noise = noise[start : start + speech_len]

        # --- Calculate RMS and Scaling Factor ---
        speech_rms = calculate_rms(speech)
        noise_rms = calculate_rms(noise)

        # Add small epsilon to prevent division by zero if a segment is pure silence
        epsilon = 1e-10
        scaling_factor = calculate_scaling_factor(speech_rms, noise_rms, target_snr_db, epsilon)

        if scaling_factor is None: # Handle potential issues from calculate_scaling_factor
             print(f"Warning: Could not calculate scaling factor for {Path(speech_path).name} and {Path(noise_path).name}. Skipping.")
             return None

        # --- Mix Audio ---
        scaled_noise = noise * scaling_factor
        mixed_audio = speech + scaled_noise

        # --- Prevent Clipping ---
        max_amp = np.max(np.abs(mixed_audio))
        if max_amp > 1.0:
            mixed_audio = mixed_audio / max_amp # Normalize *only* if clipping occurs

        # --- Prepare Output ---
        speech_basename = Path(speech_path).stem
        noise_basename = Path(noise_path).stem
        output_filename = f"{speech_basename}_mix_{noise_basename}_snr{target_snr_db}dB.wav"

        # Create SNR-specific subdirectory
        snr_output_dir = output_dir / f"SNR_{target_snr_db}dB"
        snr_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = snr_output_dir / output_filename

        # --- Save Mixed Audio ---
        sf.write(output_path, mixed_audio, samplerate, subtype='PCM_16') # Save as 16-bit WAV

        return output_path

    except Exception as e:
        print(f"Error processing {speech_path} with {noise_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Mix LibriSpeech with MUSAN/ESC-50 noise at specific SNRs.")
    parser.add_argument("--librispeech_dir", type=str, required=True, help="Path to the root directory of the LibriSpeech dataset.")
    parser.add_argument("--noise_dir", type=str, required=True, help="Path to the root directory of the noise dataset (MUSAN or ESC-50).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the mixed audio files.")
    parser.add_argument("--target_snrs", type=int, nargs='+', default=[10, 20, 30, 40], help="List of target SNRs in dB.")
    parser.add_argument("--num_files", type=int, default=None, help="Optional: Process only the first N LibriSpeech files.")
    parser.add_argument("--speech_format", type=str, default="flac", help="Format of speech files (e.g., 'flac', 'wav').")
    parser.add_argument("--noise_format", type=str, default="wav", help="Format of noise files (e.g., 'wav').")


    args = parser.parse_args()

    librispeech_dir = Path(args.librispeech_dir)
    noise_dir = Path(args.noise_dir)
    output_dir = Path(args.output_dir)
    target_snrs = args.target_snrs

    # --- Validate Paths ---
    if not librispeech_dir.is_dir():
        print(f"Error: LibriSpeech directory not found: {librispeech_dir}")
        return
    if not noise_dir.is_dir():
        print(f"Error: Noise directory not found: {noise_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Find Audio Files ---
    # Adjust glob pattern based on dataset structure and format
    # This searches recursively (**) for files with the specified extension
    speech_files = sorted(list(librispeech_dir.rglob(f'*.{args.speech_format}')))
    noise_files = sorted(list(noise_dir.rglob(f'*.{args.noise_format}')))

    if not speech_files:
        print(f"Error: No speech files (*.{args.speech_format}) found in {librispeech_dir}")
        return
    if not noise_files:
        print(f"Error: No noise files (*.{args.noise_format}) found in {noise_dir}")
        return

    print(f"Found {len(speech_files)} speech files.")
    print(f"Found {len(noise_files)} noise files.")

    if args.num_files:
        speech_files = speech_files[:args.num_files]
        print(f"Processing the first {len(speech_files)} speech files.")

    # --- Start Mixing ---
    processed_count = 0
    skipped_count = 0
    for speech_path in tqdm(speech_files, desc="Processing Speech Files"):
        if not noise_files:
            print("Warning: Ran out of noise files?") # Should not happen if list is not empty initially
            break

        # Select a random noise file for each speech file
        # If you want *different* noise for each SNR level of the *same* speech file,
        # move this line inside the SNR loop.
        noise_path = random.choice(noise_files)

        for snr in target_snrs:
            result_path = mix_audio(speech_path, noise_path, snr, output_dir)
            if result_path:
                processed_count += 1
            else:
                skipped_count += 1
                # Optional: try a different noise file if the first one failed (e.g., due to sample rate)
                # noise_path_alt = random.choice(noise_files)
                # result_path = mix_audio(speech_path, noise_path_alt, snr, output_dir) ... etc

    print("\nMixing complete.")
    print(f"Successfully created {processed_count} mixed files.")
    print(f"Skipped {skipped_count} mixing attempts (due to errors or sample rate mismatch).")

if __name__ == "__main__":
    main()