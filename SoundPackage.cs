using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace Soundscape
{
	public sealed class SoundPackage
	{
		public string FilePathFormat { get; private set; }
		
		public SoundPackage(string filePathFormat)
		{
			if (string.IsNullOrWhiteSpace(filePathFormat))
				throw new ArgumentOutOfRangeException(nameof(filePathFormat));

			FilePathFormat = filePathFormat;

			// Verify that each sound file exists

			var fileNames = Direction
				.AllDirections
				.Select(x => string.Format(FilePathFormat, x.Dir))
				.ToArray();

			var invalidFileName = 
				fileNames
				.FirstOrDefault(x => !File.Exists(x));

			if (invalidFileName != null)
				throw new FileNotFoundException(nameof(invalidFileName) + ": " + invalidFileName);
		}

		public sbyte[] GetSoundData(Direction dir)
		{
			var filePath = string.Format(FilePathFormat, (int) dir.Dir);
			using (var br = new BinaryReader(File.OpenRead(filePath)))
			{
				var header = WaveHeader.Read(br);
				var format = WaveFormat.Read(br);

				if (format.dwSamplesPerSec != 44100)
					throw new ArgumentException("Invalid sample rate: " + format.dwSamplesPerSec);
				if (format.wBitsPerSample != 16)
					throw new ArgumentException("Invalid bit depth: " + format.wBitsPerSample);
				if (format.wChannels != 2)
					throw new ArgumentException("Invalid number of channels: " + format.wChannels);

				var data = WaveDataChunk.Read(br);

				return data.shortArray;
			}
		}
	}
}
