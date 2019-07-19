using System;
using System.Linq;
using System.IO;

namespace Soundscape
{
	class Program
	{
		static void Main(string[] args)
		{
			var soundPackageSnap = new SoundPackage(@"C:\Users\Michael Vu\Desktop\stereo_samples\snap\{0}.wav");

			var numSamples = 44100 * 20 /* seconds */;
			var creator = new WaveDataCreator(numSamples);

			// Add at 1 second
			//creator.AddSound(soundPackageSnap.GetSoundData(Direction.FORWARD), 44100);

			// Add at 2 second
			//creator.AddSound(soundPackageSnap.GetSoundData(Direction.LEFT), 44100 * 2);
			//creator.AddSound(soundPackageSnap.GetSoundData(Direction.FORWARD_RIGHT), 44100 * 2 + 8000);
			var rand = new Random();
			Func<Direction> getRandomDir = () => Direction.AllDirections.Skip(rand.Next(8)).First();

			for (var i = 0; i < numSamples; i += 44100 / 4)
			{
				creator.AddSound(soundPackageSnap.GetSoundData(getRandomDir()), i);
			}

			var d = creator.Data;

			var outputHeader = new WaveHeader();
			var outputFormat = new WaveFormat();
			var outputData = new WaveDataChunk();

			outputData.shortArray = d;
			outputData.dwChunkSize = (uint) outputData.shortArray.Length;

			var outputFilePath = @"C:\Users\Michael Vu\Desktop\stereo_samples\test.wav";
			using (var bw = new BinaryWriter(File.OpenWrite(outputFilePath)))
			{
				outputHeader.Write(bw);
				outputFormat.Write(bw);
				outputData.Write(bw);
			}
		}
	}
}
