using System;
using System.Collections.Generic;
using System.Text;

namespace Soundscape
{
	public sealed class WaveDataCreator
	{
		private int _numSamples;

		public sbyte[] Data { get; }

		public WaveDataCreator(int numSamples)
		{
			if (numSamples <= 0)
				throw new ArgumentOutOfRangeException(nameof(numSamples));
			_numSamples = numSamples;

			// Stereo audio
			Data = new sbyte[_numSamples * 2];
		}

		private static sbyte AddShortsLimited(sbyte a, sbyte b)
		{
			unsafe
			{
				return (sbyte) Math.Min(Math.Max(((int)a) + ((int)b), (int) sbyte.MinValue), (int)sbyte.MaxValue);
			}
		}

		public void AddSound(sbyte[] soundData, int startTimeSample)
		{
			if (soundData == null)
				throw new ArgumentNullException(nameof(soundData));
			if (startTimeSample < 0)
				throw new ArgumentOutOfRangeException(nameof(startTimeSample));
			if (startTimeSample >= _numSamples)
				throw new ArgumentOutOfRangeException(nameof(startTimeSample));

			// 2 channels per sample
			var startIndex = startTimeSample * 2;
			
			var endIndex = Math.Min(startIndex + soundData.Length, _numSamples * 2);

			for (var i = startIndex; i < endIndex; i++)
				Data[i] = AddShortsLimited(Data[i], soundData[i - startIndex]);
		}
	}
}
