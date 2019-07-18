using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Soundscape
{
	public sealed class WaveFormat
	{
		/// <summary>
		/// Four bytes: "fmt "
		/// </summary>
		public string sChunkID { get; set; }

		/// <summary>
		/// Length of header in bytes
		/// </summary>
		public uint dwChunkSize { get; set; }

		/// <summary>
		/// 1 (MS PCM)
		/// </summary>
		public ushort wFormatTag { get; set; }

		/// <summary>
		/// Number of channels
		/// </summary>
		public ushort wChannels { get; set; }

		/// <summary>
		/// Frequency of the audio in Hz
		/// 44100
		/// </summary>
		public uint dwSamplesPerSec { get; set; }

		/// <summary>
		/// For estimating RAM allocation
		/// </summary>
		public uint dwAvgBytesPerSec { get; set; }

		/// <summary>
		/// Sample frame size, in bytes
		/// </summary>
		public ushort wBlockAlign { get; set; }

		/// <summary>
		/// Bits per sample
		/// </summary>
		public ushort wBitsPerSample { get; set; }

		public WaveFormat()
		{
			sChunkID = "fmt ";
			dwChunkSize = 16;
			wFormatTag = 1;
			wChannels = 2;
			dwSamplesPerSec = 44100;
			wBitsPerSample = 16;
			wBlockAlign = (ushort) (wChannels * (wBitsPerSample / 8));
			dwAvgBytesPerSec = dwSamplesPerSec * wBlockAlign;
		}

		public void Write(BinaryWriter bw)
		{
			bw.Write(sChunkID.ToCharArray());
			bw.Write(dwChunkSize);
			bw.Write(wFormatTag);
			bw.Write(wChannels);
			bw.Write(dwSamplesPerSec);
			bw.Write(dwAvgBytesPerSec);
			bw.Write(wBlockAlign);
			bw.Write(wBitsPerSample);
		}

		public static WaveFormat Read(BinaryReader br)
		{
			var fmt = new WaveFormat();

			fmt.sChunkID = new string(br.ReadChars(4));
			fmt.dwChunkSize = br.ReadUInt32();
			fmt.wFormatTag = br.ReadUInt16();
			fmt.wChannels = br.ReadUInt16();
			fmt.dwSamplesPerSec = br.ReadUInt32();
			fmt.dwAvgBytesPerSec = br.ReadUInt32();
			fmt.wBlockAlign = br.ReadUInt16();
			fmt.wBitsPerSample = br.ReadUInt16();

			return fmt;
		}
	}
}
