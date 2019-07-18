using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Soundscape
{
	public sealed class WaveDataChunk
	{
		public string sChunkID;
		public uint dwChunkSize;
		public short[] shortArray;

		public WaveDataChunk()
		{
			shortArray = new short[0];
			dwChunkSize = 0;
			sChunkID = "data";
		}

		public void Write(BinaryWriter br)
		{
			br.Write(sChunkID.ToCharArray());
			br.Write(dwChunkSize);
			foreach (short x in shortArray)
				br.Write(x);
		}

		public static WaveDataChunk Read(BinaryReader br)
		{
			var dataChunk = new WaveDataChunk();

			dataChunk.sChunkID = new string(br.ReadChars(4));
			dataChunk.dwChunkSize =  br.ReadUInt32();
			
			// Assume 16 bit per sample
			var dataArrLength = dataChunk.dwChunkSize / 2;
			dataChunk.shortArray = new short[dataArrLength];

			for (uint i = 0; i < dataArrLength; i++)
				dataChunk.shortArray[i] = br.ReadInt16();

			return dataChunk;
		}
	}
}
