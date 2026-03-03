## learing how to use argparse

import math
import argparse

parser = argparse.ArgumentParser(description="calculate the volume of a cylineer")
parser.add_argument("-r", "--radius", type=int, metavar='', required=True, help="radius of cylinder")
parser.add_argument("-H", "--height", type=int, metavar='', required=True, help="height of cylinder")

# mutually exclusive arguments 
group = parser.add_mutually_exclusive_group()
group.add_argument("-q", "--quiet", action="store_true", help="quiet printing")
group.add_argument("-v", "--verbose", action="store_true", help="verbose printing")


args = parser.parse_args()


# metavar='' makes the help menu look nicer
# required=True makes the argument required
# help=' ' gives the user info about the file and its arguments

def cylinder_volume(radius, height):
    vol = (math.pi) * (radius ** 2) * height
    return vol

if __name__ == "__main__":
   volume = cylinder_volume(args.radius, args.height)
   if args.quiet:
      print(volume)
   elif args.verbose:
      print(f"The volume of a cylinder with radius {args.radius} and height {args.height} is {volume}")
   else:
      print(f"Volume of cylinder: {volume}")