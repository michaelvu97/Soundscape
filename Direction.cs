using System;
using System.Collections.Generic;
using System.Text;

namespace Soundscape
{
	public struct Direction
	{
		public byte Dir
		{
			get => _direction;
			private set {
				if (value > 7)
					throw new ArgumentOutOfRangeException(nameof(Direction));
				_direction = value;
			}
		}
		private byte _direction;

		public static readonly Direction FORWARD = new Direction(0);
		public static readonly Direction FORWARD_LEFT = new Direction(1);
		public static readonly Direction LEFT = new Direction(2);
		public static readonly Direction BACKWARD_LEFT = new Direction(3);
		public static readonly Direction BACKWARD = new Direction(4);
		public static readonly Direction BACKWARD_RIGHT = new Direction(5);
		public static readonly Direction RIGHT = new Direction(6);
		public static readonly Direction FORWARD_RIGHT = new Direction(7);

		public static readonly IReadOnlyCollection<Direction> AllDirections = 
			new List<Direction>
			{
				FORWARD,
				FORWARD_LEFT,
				LEFT,
				BACKWARD_LEFT,
				BACKWARD,
				BACKWARD_RIGHT,
				RIGHT,
				FORWARD_RIGHT
			}
			.AsReadOnly();

		private Direction(byte dir)
		{
			if (dir > 7)
				throw new ArgumentOutOfRangeException(nameof(dir));
			_direction = dir;
		}
	}
}
