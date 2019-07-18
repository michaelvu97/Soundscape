using System;
using System.Collections.Generic;
using System.Text;

namespace Soundscape
{
	public sealed class WaveDataCreator
	{
		private int _numSamples;

		private short[] _data;

		public WaveDataCreator(int numSamples)
		{
			if (numSamples <= 0)
				throw new ArgumentOutOfRangeException(nameof(numSamples));
			_numSamples = numSamples;

			// Stereo audio
			_data = new short[_numSamples * 2];
		}

		private static short AddShortsLimited(short a, short b)
		{
			unsafe
			{
				return (short) Math.Min(Math.Max(((int)a) + ((int)b), (int) short.MinValue), (int)short.MaxValue);
			}
		}

		public void AddSound(short[] soundData, Direction dir, int startTimeSample)
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
				_data[i] = AddShortsLimited(_data[i], soundData[i]);
		}
	}
}
