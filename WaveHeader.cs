using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Soundscape
{
	public sealed class WaveHeader
	{
		public string sGroupID { get; set; }
		public uint dwFileLength { get; set; }
		public string sRiffType { get; set; }

		public WaveHeader()
		{
			dwFileLength = 0;
			sGroupID = "RIFF";
			sRiffType = "WAVE";
		}

		public void Write(BinaryWriter bw)
		{
			bw.Write(sGroupID.ToCharArray());
			bw.Write(dwFileLength);
			bw.Write(sRiffType.ToCharArray());
		}

		public static WaveHeader Read(BinaryReader br)
		{
			var header = new WaveHeader();

			header.sGroupID = new string(br.ReadChars(4));
			header.dwFileLength = br.ReadUInt32();
			header.sRiffType = new string(br.ReadChars(4));

			return header;
		}
	}
}
