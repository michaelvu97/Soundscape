using System;

namespace Soundscape
{
	class Program
	{
		static void Main(string[] args)
		{
			Console.WriteLine("Hello World!");
			var creator = new WaveDataCreator(44100);
		}
	}
}
