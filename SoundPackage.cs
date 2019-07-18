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
				.FirstOrDefault(x => !File.Exists(filePathFormat));

			if (invalidFileName != null)
				throw new FieldAccessException(nameof(invalidFileName) + ": " + invalidFileName);

			// 
		}

		public short[] GetSoundData(Direction dir)
		{
			var filePath = string.Format(FilePathFormat, (int) dir.Dir);
			using (var br = new BinaryReader(File.OpenRead(filePath)))
			{
				var header = WaveHeader.Read(br);
				var format = WaveFormat.Read(br);
				var data = WaveDataChunk.Read(br);

				return data.shortArray;
			}
		}
	}
}
