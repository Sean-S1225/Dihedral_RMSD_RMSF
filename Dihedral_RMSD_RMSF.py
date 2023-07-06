import numpy as np
import math
import pandas as pd

class Algorithm:
	def __init__(self, files: list[str], references: list[str], names: list[str]) -> None:
		"""
		Args:
			files (list[str]): Files containing the dihedral angle data, columns are expected to be in Phi:x, Psi:x, Phi:x+1, Psi:x+1, etc. form
			references (list[str]): Files containing the reference structures to use in calculating RMSD
			names (list[str]): File names for each RMSD dataset
			lengths (list[str]): Files containing the lengths of the dihedrals
			restrictToDBD (bool, optional): Should the RMSD be of the whole protein, or DBD only? Defaults to False.
		"""
		self.files = files
		self.references = references
		self.names = names
		self.flag = "!#!#"

	def angleBetween(self, start: float, end: float) -> float:
		"""
		Calculates the angle between two angles in degrees, returns an angle in degrees

		Args:
			start (float): Start angle, in degrees
			end (float): End angle, in degrees

		Returns:
			float: Angle between the two angles, in degrees
		"""
		abs_dist = math.fabs(start - end)
		if abs_dist > 180:
			return (360) - abs_dist
		return abs_dist

	def arcLength(self, angle: float, r: float) -> float:
		"""
		Returns the length of an arc of a circle, assuming angle is in degrees

		Args:
			angle (float): angle, in degrees
			r (float): Radius of circle

		Returns:
			float: Arclength defined by the start and end angles, and the radius
		"""
		return 2 * np.pi * r * (angle/360)

	def calcDistWraparound(self, fix: list[float], other: list[float]):
		"""
		Calculates the smallest distance between two points in a "wraparound space", with x in [-180, 180], y in [-180, 180].
		Distance is calculated by fixing one point, and then imaging the other point eight times (for a total of nine) by adding
		either -360, 0, 360 to the x and y positions, then taking the distance to each point and returning the minimum value.
		The equation is symmetric, so which point is the "fixed point" doesn't matter

		Args:
			fix_x (float): The x component of the fixed point
			fix_y (float): The y component of the fixed point
			x (float): The x component of the point to be imaged
			y (float): The y component of the point to be imaged

		Returns:
			float: The minimum distance from one point to the other
		"""
		# print(fix, other)
		min_x, x_coord = math.inf, None
		min_y, y_coord = math.inf, None
		#We can identify the imaged point with the closest distance to the fixed point one component at a time
		for i in [-360, 0, 360]:
			temp = math.fabs(fix[0] - (other[0] + i))
			if temp < min_x:
				min_x = temp
				x_coord = other[0] + i
		for i in [-360, 0, 360]:
			temp = math.fabs(fix[1] - (other[1] + i))
			if temp < min_y:
				min_y = temp
				y_coord = other[1] + i

		#Now that we've found which imaged point is closest to the fixed point, return the standard Euclidean distance to it
		return math.sqrt(pow(x_coord - fix[0], 2) + pow(y_coord - fix[1], 2))/360
	
	def writeToFile(dataFrame: pd.DataFrame, fileName: str, header: bool = False) -> None:
		"""Takes a DataFrame and writes it to a csv file

		Args:
			dataFrame (pd.DataFrame): The DataFrame to write
			name (str): The name of the file
			header (bool, optional): Should the header be included? Defaults to False.
		"""
		dataFrame.to_csv(fileName, header = header)

class ArcLength(Algorithm):
	def __init__(self, files: list[str], references: list[str], names: list[str], lengths: list[str]) -> None:
		"""
		Args:
			files (list[str]): Files containing the dihedral angle data, columns are expected to be in Phi:x, Psi:x, Phi:x+1, Psi:x+1, etc. form
			references (list[str]): Files containing the reference structures to use in calculating RMSD
			names (list[str]): File names for each RMSD dataset
			lengths (list[str]): Files containing the lengths of the dihedrals
			restrictToDBD (bool, optional): Should the RMSD be of the whole protein, or DBD only? Defaults to False.
		"""
		super().__init__(files, references, names)
		self.lengths = lengths

	def RMSD(self, restrictToDBD = False, separatePhiPsi = True):
		"""
		Calculates the Arclength RMSD values of a set of dihedral angles, and writes them to a file

		Args:
			files (list[str]): Files containing the dihedral angle data, columns are expected to be in Phi:x, Psi:x, Phi:x+1, Psi:x+1, etc. form
			references (list[str]): Files containing the reference structures to use in calculating RMSD
			names (list[str]): File names for each RMSD dataset
			lengths (list[str]): Files containing the lengths of the dihedrals
			restrictToDBD (bool, optional): Should the RMSD be of the whole protein, or DBD only? Defaults to False.
		"""
		
		toReturn = []
		for file, ref, name, length in zip(self.files, self.references, self.names, self.lengths):
			angles = pd.read_csv(file, delim_whitespace=True)
			angles = angles.drop(angles.columns[[0]], axis = 1)

			ref = pd.read_csv(ref, delim_whitespace=True)
			ref = ref.drop(ref.columns[[0]], axis=1)[:1]

			length = pd.read_csv(length, delim_whitespace=True)
			length = length.drop(length.columns[0], axis=1)

			if restrictToDBD:
				#Restricts the range to only the DBD (134-312)
				angles = angles.drop(angles.columns[2*311:], axis = 1)
				angles = angles.drop(angles.columns[:132*2], axis = 1)

				ref = ref.drop(ref.columns[2*311:], axis = 1)
				ref = ref.drop(ref.columns[:132*2], axis = 1)

				length = length.drop(length.columns[312:], axis = 1)
				length = length.drop(length.columns[:132], axis = 1)

			#Drop every other column to isolate the DataFrame to be only phi or psi angles
			anglesPhi = angles.drop(angles.columns[list(range(len(angles.columns) + 1))[1::2]], axis = 1)[:1000]
			anglesPsi = angles.drop(angles.columns[list(range(len(angles.columns)))[0::2]], axis = 1)[:1000]

			refPhi = ref.drop(ref.columns[list(range(len(ref.columns) + 1))[1::2]], axis = 1)
			refPsi = ref.drop(ref.columns[list(range(len(ref.columns)))[0::2]], axis = 1)

			refPhi = pd.concat([refPhi] * anglesPhi.shape[0])
			length = pd.concat([length] * anglesPhi.shape[0])

			if separatePhiPsi:
				temp = pd.DataFrame(np.vectorize(lambda x, y: self.arcLength(x, y))(pd.DataFrame(np.vectorize(self.angleBetween)(refPhi, anglesPhi)), length.drop(length.columns[-1], axis=1))).apply(np.square).mean(axis=1).apply(np.sqrt)
				toReturn.append(temp)
				temp.to_csv(name.replace(self.flag, "Phi"), header=False)
				temp = pd.DataFrame(np.vectorize(lambda x, y: self.arcLength(x, y))(pd.DataFrame(np.vectorize(self.angleBetween)(refPsi, anglesPsi)), length.drop(length.columns[0], axis=1))).apply(np.square).mean(axis=1).apply(np.sqrt)
				toReturn.append(temp)
				temp.to_csv(name.replace(self.flag, "Psi"), header=False)
			else:
				temp = pd.concat([pd.DataFrame(np.vectorize(lambda x, y: self.arcLength(x, y))(pd.DataFrame(np.vectorize(self.angleBetween)(refPhi, anglesPhi)), length.drop(length.columns[-1], axis=1))),
	       				          pd.DataFrame(np.vectorize(lambda x, y: self.arcLength(x, y))(pd.DataFrame(np.vectorize(self.angleBetween)(refPsi, anglesPsi)), length.drop(length.columns[0], axis=1)))], axis=1).apply(np.square).mean(axis=1).apply(np.sqrt)
				toReturn.append(temp)
				temp.to_csv(name, header=False)

			return toReturn
		
	def RMSF(self):
		"""
		Calculates the Arclength RMSF values of a set of dihedral angles, and writes them to a file

		Args:
			files (list[str]): Files containing the dihedral angle data, columns are expected to be in Phi:x, Psi:x, Phi:x+1, Psi:x+1, etc. form
			references (list[str]): Files containing the reference structures to use in calculating RMSD
			names (list[str]): File names for each RMSD dataset
			lengths (list[str]): Files containing the lengths of the dihedrals
			restrictToDBD (bool, optional): Should the RMSD be of the whole protein, or DBD only? Defaults to False.
		"""
		
		for file, ref, name, length in zip(self.files, self.references, self.names, self.lengths):
			angles = pd.read_csv(file, delim_whitespace=True)
			angles = angles.drop(angles.columns[[0]], axis = 1)

			anglesPhi = angles.drop(angles.columns[list(range(len(angles.columns) + 1))[1::2]], axis = 1)[:1000]
			anglesPsi = angles.drop(angles.columns[list(range(len(angles.columns)))[0::2]], axis = 1)[:1000]

			ref = pd.read_csv(ref, delim_whitespace=True)
			ref = ref.drop(ref.columns[[0]], axis=1)

			refPhi = ref.drop(ref.columns[list(range(len(ref.columns) + 1))[1::2]], axis = 1)
			refPsi = ref.drop(ref.columns[list(range(len(ref.columns)))[0::2]], axis = 1)

			length = pd.read_csv(length, delim_whitespace=True)
			length = length.drop(length.columns[0], axis=1)

			refPhi = pd.concat([refPhi] * anglesPhi.shape[0])
			length = pd.concat([length] * anglesPhi.shape[0])

			pd.DataFrame(np.vectorize(self.arcLength)(pd.DataFrame(np.vectorize(self.angleBetween)(anglesPhi, refPhi)), length.drop(length.columns[-1], axis=1))).apply(np.square).mean(axis=0).apply(np.sqrt).to_csv(name.replace(self.flag, "Phi"), header=False)
			pd.DataFrame(np.vectorize(self.arcLength)(pd.DataFrame(np.vectorize(self.angleBetween)(anglesPsi, refPsi)), length.drop(length.columns[0], axis=1))).apply(np.square).mean(axis=0).apply(np.sqrt).to_csv(name.replace(self.flag, "Psi"), header=False)


class AngleDifference(Algorithm):
	def __init__(self, files: list[str], references: list[str], names: list[str]) -> None:
		"""
		Args:
			files (list[str]): Files containing the dihedral angle data, columns are expected to be in Phi:x, Psi:x, Phi:x+1, Psi:x+1, etc. form
			references (list[str]): Files containing the reference structures to use in calculating RMSD
			names (list[str]): File names for each RMSD dataset
			lengths (list[str]): Files containing the lengths of the dihedrals
			restrictToDBD (bool, optional): Should the RMSD be of the whole protein, or DBD only? Defaults to False.
		"""
		super().__init__(files, references, names)

	def RMSD(self, restrictToDBD = False, separatePhiPsi = True):
		"""
		Calculates the Angle Difference RMSD values of a set of dihedral angles, and writes them to a file

		Args:
			files (list[str]): Files containing the dihedral angle data, columns are expected to be in Phi:x, Psi:x, Phi:x+1, Psi:x+1, etc. form
			references (list[str]): Files containing the reference structures to use in calculating RMSD
			names (list[str]): File names for each RMSD dataset
			restrictToDBD (bool, optional): Should the RMSD be of the whole protein, or DBD only? Defaults to False.
		"""
		for file, ref, name in zip(self.files, self.references, self.names):
			angles = pd.read_csv(file, delim_whitespace=True)
			angles = angles.drop(angles.columns[[0]], axis = 1)[:1000]

			ref = pd.read_csv(ref, delim_whitespace=True)
			ref = ref.drop(ref.columns[[0]], axis=1)[:1]

			if restrictToDBD:
				#Restricts the range to only the DBD (134-312)
				angles = angles.drop(angles.columns[2*311:], axis = 1)
				angles = angles.drop(angles.columns[:132*2], axis = 1)

				ref = ref.drop(ref.columns[2*311:], axis = 1)
				ref = ref.drop(ref.columns[:132*2], axis = 1)

			#Drop every other column to isolate the DataFrame to be only phi or psi angles
			anglesPhi = angles.drop(angles.columns[list(range(len(angles.columns) + 1))[1::2]], axis = 1)[:1000]
			anglesPsi = angles.drop(angles.columns[list(range(len(angles.columns)))[0::2]], axis = 1)[:1000]

			#Drop every other column to isolate the DataFrame to be only phi or psi angles
			refPhi = ref.drop(ref.columns[list(range(len(ref.columns) + 1))[1::2]], axis = 1)
			refPsi = ref.drop(ref.columns[list(range(len(ref.columns)))[0::2]], axis = 1)

			refPhi = pd.concat([refPhi] * anglesPhi.shape[0])
			refPsi = pd.concat([refPsi] * anglesPsi.shape[0])

			if separatePhiPsi:
				pd.DataFrame(np.vectorize(self.angleBetween)(refPhi, anglesPhi)).apply(lambda x: x/360).apply(np.square).mean(axis=1).apply(np.sqrt).to_csv(name.replace(self.flag, "Phi"), header=False)
				pd.DataFrame(np.vectorize(self.angleBetween)(refPsi, anglesPsi)).apply(lambda x: x/360).apply(np.square).mean(axis=1).apply(np.sqrt).to_csv(name.replace(self.flag, "Psi"), header=False)
			else:
				pd.concat([pd.DataFrame(np.vectorize(self.angleBetween)(refPhi, anglesPhi)), pd.DataFrame(np.vectorize(self.angleBetween)(refPsi, anglesPsi))], axis=1).apply(lambda x: x/360).apply(np.square).mean(axis=1).apply(np.sqrt).to_csv(name, header=False)

	def PairwiseRMSD(self, restrictToDBD = False, files = None, references = None, name = None):
		if files == None: files = self.files
		if references == None: references = self.references
		if name == None: name = self.names

		fileList = []
		for file in files:
			file = pd.read_csv(file, delim_whitespace=True)
			file = file.drop(file.columns[[0]], axis=1)
			fileList.append(file)

		referenceList = []
		for ref in references:
			ref = pd.read_csv(ref, delim_whitespace=True)
			ref = ref.drop(ref.columns[[0]], axis=1)
			referenceList.append(ref)

		angles = pd.concat(fileList)
		ref = pd.concat(referenceList)

		angles = angles.reset_index()
		ref = ref.reset_index()

		if restrictToDBD:
			#Restricts the range to only the DBD (134-312)
			angles = angles.drop(angles.columns[2*311:], axis = 1)
			angles = angles.drop(angles.columns[:132*2], axis = 1)

			ref = ref.drop(ref.columns[2*311:], axis = 1)
			ref = ref.drop(ref.columns[:132*2], axis = 1)

		#Drop every other column to isolate the DataFrame to be only phi or psi angles
		anglesPhi = angles.drop(angles.columns[list(range(len(angles.columns) + 1))[1::2]], axis = 1)
		anglesPsi = angles.drop(angles.columns[list(range(len(angles.columns)))[0::2]], axis = 1)

		refPhi = ref.drop(ref.columns[list(range(len(ref.columns) + 1))[1::2]], axis = 1)
		refPsi = ref.drop(ref.columns[list(range(len(ref.columns)))[0::2]], axis = 1)

		matrix = []

		num = angles.shape[0]
		for x in range(num):
			currPhi = pd.concat([refPhi.loc[[x]]] * anglesPhi.shape[0])
			currPsi = pd.concat([refPsi.loc[[x]]] * anglesPhi.shape[0])
			
			matrix.append(pd.concat([pd.DataFrame(np.vectorize(self.angleBetween)(currPhi, anglesPhi)), pd.DataFrame(np.vectorize(self.angleBetween)(currPsi, anglesPsi))], axis=1).apply(lambda x: x/360).apply(np.square).mean(axis=1).apply(np.sqrt))

			if x % 100 == 0:
				print(f"{x}/{num} Completed")

		matrix = pd.concat(matrix, axis=1)

		matrix.to_csv(name, header = False, index = False)

	def RMSF(self):
		"""
		Calculates the Angle Difference RMSF values of a set of dihedral angles, and writes them to a file

		Args:
			files (list[str]): Files containing the dihedral angle data, columns are expected to be in Phi:x, Psi:x, Phi:x+1, Psi:x+1, etc. form
			references (list[str]): Files containing the reference structures to use in calculating RMSD
			names (list[str]): File names for each RMSD dataset
			lengths (list[str]): Files containing the lengths of the dihedrals
			restrictToDBD (bool, optional): Should the RMSD be of the whole protein, or DBD only? Defaults to False.
		"""

		for file, ref, name in zip(self.files, self.references, self.names):
			angles = pd.read_csv(file, delim_whitespace=True)
			angles = angles.drop(angles.columns[[0]], axis = 1)

			anglesPhi = angles.drop(angles.columns[list(range(len(angles.columns) + 1))[1::2]], axis = 1)[:1000]
			anglesPsi = angles.drop(angles.columns[list(range(len(angles.columns)))[0::2]], axis = 1)[:1000]

			ref = pd.read_csv(ref, delim_whitespace=True)
			ref = ref.drop(ref.columns[[0]], axis=1)

			refPhi = ref.drop(ref.columns[list(range(len(ref.columns) + 1))[1::2]], axis = 1)
			refPsi = ref.drop(ref.columns[list(range(len(ref.columns)))[0::2]], axis = 1)

			refPhi = pd.concat([refPhi] * anglesPhi.shape[0])
			refPsi = pd.concat([refPsi] * anglesPsi.shape[0])

			pd.DataFrame(np.vectorize(self.angleBetween)(anglesPhi, refPhi)).apply(lambda x: x/360).apply(np.square).mean(axis=0).apply(np.sqrt).to_csv(name.replace(self.flag, "Phi"), header=False)
			pd.DataFrame(np.vectorize(self.angleBetween)(anglesPsi, refPsi)).apply(lambda x: x/360).apply(np.square).mean(axis=0).apply(np.sqrt).to_csv(name.replace(self.flag, "Psi"), header=False)

class Wraparound(Algorithm):
	def __init__(self, files: list[str], references: list[str], names: list[str]) -> None:
		"""
		Args:
			files (list[str]): Files containing the dihedral angle data, columns are expected to be in Phi:x, Psi:x, Phi:x+1, Psi:x+1, etc. form
			references (list[str]): Files containing the reference structures to use in calculating RMSD
			names (list[str]): File names for each RMSD dataset
			lengths (list[str]): Files containing the lengths of the dihedrals
			restrictToDBD (bool, optional): Should the RMSD be of the whole protein, or DBD only? Defaults to False.
		"""
		super().__init__(files, references, names)

	def RMSD(self, restrictToDBD = False):
		"""
		Calculates the Wraparound RMSD values of a set of dihedral angles, and writes them to a file

		Args:
			files (list[str]): Files containing the dihedral angle data, columns are expected to be in Phi:x, Psi:x, Phi:x+1, Psi:x+1, etc. form
			references (list[str]): Files containing the reference structures to use in calculating RMSD
			names (list[str]): File names for each RMSD dataset
			restrictToDBD (bool, optional): Should the RMSD be of the whole protein, or DBD only? Defaults to False.
		"""
		for file, ref, name in zip(self.files, self.references, self.names):
			angles = pd.read_csv(file, delim_whitespace=True)
			angles = angles.drop(angles.columns[[0]], axis = 1)[:1000]

			ref = pd.read_csv(ref, delim_whitespace=True)
			ref = ref.drop(ref.columns[[0]], axis=1)[:1]

			if restrictToDBD:
				#Restricts the range to only the DBD (134-312)
				angles = angles.drop(angles.columns[2*311:], axis = 1)
				angles = angles.drop(angles.columns[:132*2], axis = 1)

				ref = ref.drop(ref.columns[2*311:], axis = 1)
				ref = ref.drop(ref.columns[:132*2], axis = 1)

			temp = len(angles.columns)
			for x in range(len(angles.columns)//2):
				ref[str(x)] = ref.iloc[:, [2*x, 2*x+1]].apply(lambda x: [x[0], x[1]], axis=1)
				angles[str(x)] = angles.iloc[:, [2*x, 2*x+1]].apply(lambda x: [x[0], x[1]], axis=1)
			ref = ref.drop(ref.columns[:temp], axis=1)
			angles = angles.drop(angles.columns[:temp], axis=1)
			ref = pd.concat([ref] * angles.shape[0])
			pd.DataFrame(np.vectorize(self.calcDistWraparound)(ref, angles)).apply(np.square).mean(axis=1).apply(np.sqrt).to_csv(name, header=False)

	def RMSF(self):
		"""
		Calculates the Wraparound RMSF values of a set of dihedral angles, and writes them to a file

		Args:
			files (list[str]): Files containing the dihedral angle data, columns are expected to be in Phi:x, Psi:x, Phi:x+1, Psi:x+1, etc. form
			references (list[str]): Files containing the reference structures to use in calculating RMSD
			names (list[str]): File names for each RMSD dataset
			lengths (list[str]): Files containing the lengths of the dihedrals
			restrictToDBD (bool, optional): Should the RMSD be of the whole protein, or DBD only? Defaults to False.
		"""

		for file, ref, name in zip(self.files, self.references, self.names):
			angles = pd.read_csv(file, delim_whitespace=True)
			angles = angles.drop(angles.columns[[0]], axis = 1)[:1000]

			ref = pd.read_csv(ref, delim_whitespace=True)
			ref = ref.drop(ref.columns[[0]], axis=1)[:1]

			temp = len(angles.columns)
			for x in range(len(angles.columns)//2):
				ref[str(x)] = ref.iloc[:, [2*x, 2*x+1]].apply(lambda x: [x[0], x[1]], axis=1)
				angles[str(x)] = angles.iloc[:, [2*x, 2*x+1]].apply(lambda x: [x[0], x[1]], axis=1)
			ref = ref.drop(ref.columns[:temp], axis=1)
			angles = angles.drop(angles.columns[:temp], axis=1)
			ref = pd.concat([ref] * angles.shape[0])

			pd.DataFrame(np.vectorize(self.calcDistWraparound)(ref, angles)).apply(np.square).mean(axis=0).apply(np.sqrt).to_csv(name, header=False)

