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
			var invalidFileName = 
				Direction
				.AllDirections
				.Select(x => string.Format(FilePathFormat, x.Dir))
				.FirstOrDefault(x => !File.Exists(filePathFormat));

			if (invalidFileName != null)
				throw new FieldAccessException(nameof(invalidFileName) + ": " + invalidFileName);
		}

		public object GetSound(Direction dir)
		{
			var filePath = string.Format(FilePathFormat, (int) dir.Dir);
			
		}
	}
}
