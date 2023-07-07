import numpy as np
import math
import pandas as pd
import csv

#TODO: Test implementation of Angle Diff pairwise matrix
#TODO: Test implementation of Arc Length pairwise matrix
#TODO: Test implementation of Wraparound pairwise matrix

class Algorithm:
	def __init__(self, files: list[str], references: list[str], names: list[str], header: bool = False) -> None:
		"""
		Args:
			files (list[str]): Files containing the dihedral angle data, columns are expected to be in Phi:x, Psi:x, Phi:x+1, Psi:x+1, etc. form
			references (list[str]): Files containing the reference structures to use in calculating RMSD
			names (list[str]): File names for each RMSD dataset
			header (bool): Do your files have a header?
		"""
		self.files = files
		self.references = references
		self.names = names
		self.flag = "!#!#"
		if header == True:
			self.header = "infer"
		else:
			self.header = None

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
		Calculates the smallest distance between two points in a "wraparound space", with each component in [-180, 180].
		Distance is calculated by fixing one point, and then imaging the other point 3^N times, where N is the number of dimensions, by adding
		either -360, 0, 360 to each component, then taking the distance to each point and returning the minimum value.
		The equation is symmetric, so which point is the "fixed point" doesn't matter.

		Args:
			fix (list[float]): The components of the fixed point
			other (list[float]): The components of the point to be imaged

		Returns:
			float: The minimum distance from one point to the other
		"""

		if len(fix) != len(other):
			raise ValueError("The lengths of the two lists should be the same.")

		minComponents = [math.inf] * len(fix)
		coords = [None] * len(fix)

		#We can identify the imaged point with the closest distance to the fixed point one component at a time
		for i in range(len(fix)):
			for offset in [-360, 0, 360]:
				temp = math.fabs(fix[i] - (other[i] + offset))
				if temp < minComponents[i]:
					minComponents[i] = temp
					coords[i] = other[i] + offset

		#Now that we've found which imaged point is closest to the fixed point, return the standard Euclidean distance to it
		temp = 0
		for coord, fixComp in zip(coords, fix):
			temp += pow(coord - fixComp, 2)
		return math.sqrt(temp)/360

class ArcLength(Algorithm):
	def __init__(self, files: list[str], references: list[str], names: list[str], lengths: list[str], header: bool) -> None:
		"""
		Args:
			files (list[str]): Files containing the dihedral angle data, columns are expected to be in Phi:x, Psi:x, Phi:x+1, Psi:x+1, etc. form
			references (list[str]): Files containing the reference structures to use in calculating RMSD
			names (list[str]): File names for each RMSD dataset
			lengths (list[str]): Files containing the lengths of the dihedrals
			header (bool): Do your files have a header?
		"""
		super().__init__(files, references, names, header)
		self.lengths = lengths

	def RMSD(self, restrictToDBD: bool = False, separatePhiPsi: bool = True):
		"""
		Calculates the Arclength RMSD values of a set of dihedral angles, and writes them to a file

		Args:
			restrictToDBD (bool, optional): Should the RMSD be of the whole protein, or DBD only? Defaults to False.
			separatePhiPsi (bool, optional): Should there be one calculation for the Phi angles and one for the Psi angles, or should both of them be in one calculation?
		"""
		
		for file, ref, name, length in zip(self.files, self.references, self.names, self.lengths):
			#Read in files, and remove the first column (the first column contains the row numbers)
			angles = pd.read_csv(file, delim_whitespace=True, header=self.header)
			angles = angles.drop(angles.columns[[0]], axis = 1)

			ref = pd.read_csv(ref, delim_whitespace=True, header=self.header)
			ref = ref.drop(ref.columns[[0]], axis=1)[:1]

			length = pd.read_csv(length, delim_whitespace=True, header=self.header)
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
			anglesPhi = angles.drop(angles.columns[list(range(len(angles.columns) + 1))[1::2]], axis = 1)
			anglesPsi = angles.drop(angles.columns[list(range(len(angles.columns)))[0::2]], axis = 1)

			refPhi = ref.drop(ref.columns[list(range(len(ref.columns) + 1))[1::2]], axis = 1)
			refPsi = ref.drop(ref.columns[list(range(len(ref.columns)))[0::2]], axis = 1)

			refPhi = pd.concat([refPhi] * anglesPhi.shape[0])
			length = pd.concat([length] * anglesPhi.shape[0])

			#Calculate RMSD values all at once
			if separatePhiPsi:
				pd.DataFrame(np.vectorize(lambda x, y: self.arcLength(x, y))(pd.DataFrame(np.vectorize(self.angleBetween)(refPhi, anglesPhi)), length.drop(length.columns[-1], axis=1))).apply(np.square).mean(axis=1).apply(np.sqrt).to_csv(name.replace(self.flag, "Phi"), header=False)
				pd.DataFrame(np.vectorize(lambda x, y: self.arcLength(x, y))(pd.DataFrame(np.vectorize(self.angleBetween)(refPsi, anglesPsi)), length.drop(length.columns[0], axis=1))).apply(np.square).mean(axis=1).apply(np.sqrt).to_csv(name.replace(self.flag, "Psi"), header=False)
			else:
				pd.concat([pd.DataFrame(np.vectorize(lambda x, y: self.arcLength(x, y))(pd.DataFrame(np.vectorize(self.angleBetween)(refPhi, anglesPhi)), length.drop(length.columns[-1], axis=1))),
	                       pd.DataFrame(np.vectorize(lambda x, y: self.arcLength(x, y))(pd.DataFrame(np.vectorize(self.angleBetween)(refPsi, anglesPsi)), length.drop(length.columns[0], axis=1)))], axis=1).apply(np.square).mean(axis=1).apply(np.sqrt).to_csv(name, header=False)

	def PairwiseRMSD(self, restrictToDBD: bool = False):
		"""Uses the dihedral angles provided to generate a pairwise matrix of RMSD values, used for K-Means clustering

		Args:
			restrictToDBD (bool, optional): Should only the DBD be considered in the calculation? Defaults to False.
		"""

		for file, ref, name, length in zip(self.files, self.references, self.names, self.lengths):
			fileList = []
			referenceList = []
			lengthList = []

			#If file is a list of file names, then read them all in.
			#Otherwise it's only a single file, so read it in and add it to an array 
			if type(file) == list:
				for temp in self.files:
					temp = pd.read_csv(temp, delim_whitespace=True, header=self.header)
					temp = temp.drop(temp.columns[[0]], axis=1)
					fileList.append(temp)
			else:
				temp = pd.read_csv(file, delim_whitespace=True, header=self.header)
				temp = file.drop(temp.columns[[0]], axis=1)
				fileList.append(temp)

			if type(self.references[0]) == list:
				for temp in self.references:
					temp = pd.read_csv(temp, delim_whitespace=True, header=self.header)
					temp = temp.drop(temp.columns[[0]], axis=1)
					referenceList.append(temp)
			else:
				temp = pd.read_csv(ref, delim_whitespace=True, header=self.header)
				temp = temp.drop(temp.columns[[0]], axis=1)
				referenceList.append(temp)

			if type(self.lengths[0]) == list:
				for temp in self.lengths:
					temp = pd.read_csv(temp, delim_whitespace=True, header=self.header)
					temp = temp.drop(temp.columns[[0]], axis=1)
					lengthList.append(temp)
			else:
				temp = pd.read_csv(length, delim_whitespace=True, header=self.header)
				temp = temp.drop(temp.columns[[0]], axis=1)
				referenceList.append(temp)

			#Since we know that fileList and referenceList are lists (possibly of length one), we can safely do this
			angles = pd.concat(fileList)
			ref = pd.concat(referenceList)
			lens = pd.concat(lengthList)

			#Reset the index because when concatenated, each DataFrame brings it's own indexing
			angles = angles.reset_index()
			ref = ref.reset_index()
			lens = lens.reset_index()

			if restrictToDBD:
				#Restricts the range to only the DBD (134-312)
				angles = angles.drop(angles.columns[2*311:], axis = 1)
				angles = angles.drop(angles.columns[:132*2], axis = 1)

				ref = ref.drop(ref.columns[2*311:], axis = 1)
				ref = ref.drop(ref.columns[:132*2], axis = 1)

				lens = lens.drop(lens.columns[312:], axis = 1)
				lens = lens.drop(lens.columns[:132], axis = 1)

			#Drop every other column to isolate the DataFrame to be only phi or psi angles
			anglesPhi = angles.drop(angles.columns[list(range(len(angles.columns) + 1))[1::2]], axis = 1)
			anglesPsi = angles.drop(angles.columns[list(range(len(angles.columns)))[0::2]], axis = 1)

			refPhi = ref.drop(ref.columns[list(range(len(ref.columns) + 1))[1::2]], axis = 1)
			refPsi = ref.drop(ref.columns[list(range(len(ref.columns)))[0::2]], axis = 1)

			matrix = []

			num = angles.shape[0]
			for x in range(num):
				currPhi = pd.concat([refPhi.loc[[x]]] * anglesPhi.shape[0])
				currPsi = pd.concat([refPsi.loc[[x]]] * anglesPsi.shape[0])
				
				#Calculates the RMSD given the current reference Phi, Psi row
				matrix.append(pd.concat([pd.DataFrame(np.vectorize(lambda x, y: self.arcLength(x, y))(pd.DataFrame(np.vectorize(self.angleBetween)(currPhi, anglesPhi)), lens.drop(lens.columns[-1], axis=1))),
	                                     pd.DataFrame(np.vectorize(lambda x, y: self.arcLength(x, y))(pd.DataFrame(np.vectorize(self.angleBetween)(currPsi, anglesPsi)), lens.drop(lens.columns[0], axis=1)))], axis=1).apply(np.square).mean(axis=1).apply(np.sqrt))
				# matrix.append(pd.concat([pd.DataFrame(np.vectorize(self.angleBetween)(currPhi, anglesPhi)), pd.DataFrame(np.vectorize(self.angleBetween)(currPsi, anglesPsi))], axis=1).apply(lambda x: x/360).apply(np.square).mean(axis=1).apply(np.sqrt))

				if x % 100 == 0:
					print(f"{x}/{num} Completed")

			matrix = pd.concat(matrix, axis=1)

			#The pairwise matrix has been generated, but we need to convert it into a format that Amber can read
			with open(name, "w") as csvfile:
				csvwriter = csv.writer(csvfile, delimiter="\t")
				csvwriter.writerow(["#F1", "F2", "RMSD"])
				for col in range(matrix.shape[0]):
					for row, val in enumerate(matrix[col]):
						if row < col + 1:
							continue
						csvwriter.writerow([col+1, row+1, val])

	def RMSF(self):
		"""
		Calculates the Arclength RMSF values of a set of dihedral angles, and writes them to a file
		"""
		
		for file, ref, name, length in zip(self.files, self.references, self.names, self.lengths):
			#Read in files, and remove the first column (the first column contains the row numbers)
			angles = pd.read_csv(file, delim_whitespace=True, header=self.header)
			angles = angles.drop(angles.columns[[0]], axis = 1)

			anglesPhi = angles.drop(angles.columns[list(range(len(angles.columns) + 1))[1::2]], axis = 1)
			anglesPsi = angles.drop(angles.columns[list(range(len(angles.columns)))[0::2]], axis = 1)

			ref = pd.read_csv(ref, delim_whitespace=True, header=self.header)
			ref = ref.drop(ref.columns[[0]], axis=1)[:1]

			refPhi = ref.drop(ref.columns[list(range(len(ref.columns) + 1))[1::2]], axis = 1)
			refPsi = ref.drop(ref.columns[list(range(len(ref.columns)))[0::2]], axis = 1)

			length = pd.read_csv(length, delim_whitespace=True, header=self.header)
			length = length.drop(length.columns[0], axis=1)

			refPhi = pd.concat([refPhi] * anglesPhi.shape[0])
			length = pd.concat([length] * anglesPhi.shape[0])

			#Calculate RMSF values all at once
			pd.DataFrame(np.vectorize(self.arcLength)(pd.DataFrame(np.vectorize(self.angleBetween)(anglesPhi, refPhi)), length.drop(length.columns[-1], axis=1))).apply(np.square).mean(axis=0).apply(np.sqrt).to_csv(name.replace(self.flag, "Phi"), header=False)
			pd.DataFrame(np.vectorize(self.arcLength)(pd.DataFrame(np.vectorize(self.angleBetween)(anglesPsi, refPsi)), length.drop(length.columns[0], axis=1))).apply(np.square).mean(axis=0).apply(np.sqrt).to_csv(name.replace(self.flag, "Psi"), header=False)


class AngleDifference(Algorithm):
	def __init__(self, files: list[str] | list[list[str]], references: list[str] | list[list[str]], names: list[str], header: bool) -> None:
		"""
		Args:
			files (list[str] | list[list[str]]): Files containing the dihedral angle data, columns are expected to be in Phi:x, Psi:x, Phi:x+1, Psi:x+1, etc. form
			references (list[str] | list[list[str]]): Files containing the reference structures to use in calculating RMSD
			names (list[str]): File names for each RMSD dataset
			header (bool): Do your files have a header?
		"""
		super().__init__(files, references, names, header)

	def RMSD(self, restrictToDBD: bool = False, separatePhiPsi: bool = True):
		"""
		Calculates the Angle Difference RMSD values of a set of dihedral angles, and writes them to a file

		Args:
			restrictToDBD (bool, optional): Should the RMSD be of the whole protein, or DBD only? Defaults to False.
			separatePhiPsi (bool, optional): Should there be one calculation for the Phi angles and one for the Psi angles, or should both of them be in one calculation?
		"""
		for file, ref, name in zip(self.files, self.references, self.names):
			#Read in files, and remove the first column (the first column contains the row numbers)
			angles = pd.read_csv(file, delim_whitespace=True, header=self.header)
			angles = angles.drop(angles.columns[[0]], axis = 1)

			ref = pd.read_csv(ref, delim_whitespace=True, header=self.header)
			ref = ref.drop(ref.columns[[0]], axis=1)[:1]

			if restrictToDBD:
				#Restricts the range to only the DBD (134-312)
				angles = angles.drop(angles.columns[2*311:], axis = 1)
				angles = angles.drop(angles.columns[:132*2], axis = 1)

				ref = ref.drop(ref.columns[2*311:], axis = 1)
				ref = ref.drop(ref.columns[:132*2], axis = 1)

			#Drop every other column to isolate the DataFrame to be only phi or psi angles
			anglesPhi = angles.drop(angles.columns[list(range(len(angles.columns) + 1))[1::2]], axis = 1)
			anglesPsi = angles.drop(angles.columns[list(range(len(angles.columns)))[0::2]], axis = 1)

			#Drop every other column to isolate the DataFrame to be only phi or psi angles
			refPhi = ref.drop(ref.columns[list(range(len(ref.columns) + 1))[1::2]], axis = 1)
			refPsi = ref.drop(ref.columns[list(range(len(ref.columns)))[0::2]], axis = 1)

			refPhi = pd.concat([refPhi] * anglesPhi.shape[0])
			refPsi = pd.concat([refPsi] * anglesPi.shape[0])
			
			#Calculate RMSD values all at once
			if separatePhiPsi:
				pd.DataFrame(np.vectorize(self.angleBetween)(refPhi, anglesPhi)).apply(lambda x: x/360).apply(np.square).mean(axis=1).apply(np.sqrt).to_csv(name.replace(self.flag, "Phi"), header=False)
				pd.DataFrame(np.vectorize(self.angleBetween)(refPsi, anglesPsi)).apply(lambda x: x/360).apply(np.square).mean(axis=1).apply(np.sqrt).to_csv(name.replace(self.flag, "Psi"), header=False)
			else:
				pd.concat([pd.DataFrame(np.vectorize(self.angleBetween)(refPhi, anglesPhi)), pd.DataFrame(np.vectorize(self.angleBetween)(refPsi, anglesPsi))], axis=1).apply(lambda x: x/360).apply(np.square).mean(axis=1).apply(np.sqrt).to_csv(name, header=False)

	def PairwiseRMSD(self, restrictToDBD: bool = False):
		"""Uses the dihedral angles provided to generate a pairwise matrix of RMSD values, used for K-Means clustering

		Args:
			restrictToDBD (bool, optional): Should only the DBD be considered in the calculation? Defaults to False.
		"""

		for file, ref, name in zip(self.files, self.references, self.names):
			fileList = []
			referenceList = []

			#If file is a list of file names, then read them all in.
			#Otherwise it's only a single file, so read it in and add it to an array 
			if type(file) == list:
				for temp in self.files:
					temp = pd.read_csv(temp, delim_whitespace=True, header=self.header)
					temp = temp.drop(temp.columns[[0]], axis=1)
					fileList.append(temp)
			else:
				temp = pd.read_csv(file, delim_whitespace=True, header=self.header)
				temp = file.drop(temp.columns[[0]], axis=1)
				fileList.append(temp)

			#If file is a list of file names, then read them all in.
			#Otherwise it's only a single file, so read it in and add it to an array 
			if type(self.references[0]) == list:
				for temp in self.references:
					temp = pd.read_csv(temp, delim_whitespace=True, header=self.header)
					temp = temp.drop(temp.columns[[0]], axis=1)
					referenceList.append(temp)
			else:
				temp = pd.read_csv(ref, delim_whitespace=True, header=self.header)
				temp = temp.drop(temp.columns[[0]], axis=1)
				referenceList.append(temp)

			#Since we know that fileList and referenceList are lists (possibly of length one), we can safely do this
			angles = pd.concat(fileList)
			ref = pd.concat(referenceList)

			#Reset the index because when concatenated, each DataFrame brings it's own indexing
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
				currPsi = pd.concat([refPsi.loc[[x]]] * anglesPsi.shape[0])
				
				#Calculates the RMSD given the current reference Phi, Psi row
				matrix.append(pd.concat([pd.DataFrame(np.vectorize(self.angleBetween)(currPhi, anglesPhi)), pd.DataFrame(np.vectorize(self.angleBetween)(currPsi, anglesPsi))], axis=1).apply(lambda x: x/360).apply(np.square).mean(axis=1).apply(np.sqrt))

				if x % 100 == 0:
					print(f"{x}/{num} Completed")

			matrix = pd.concat(matrix, axis=1)

			#The pairwise matrix has been generated, but we need to convert it into a format that Amber can read
			with open(name, "w") as csvfile:
				csvwriter = csv.writer(csvfile, delimiter="\t")
				csvwriter.writerow(["#F1", "F2", "RMSD"])
				for col in range(matrix.shape[0]):
					for row, val in enumerate(matrix[col]):
						if row < col + 1:
							continue
						csvwriter.writerow([col+1, row+1, val])

	def RMSF(self):
		"""
		Calculates the Angle Difference RMSF values of a set of dihedral angles, and writes them to a file
		"""

		for file, ref, name in zip(self.files, self.references, self.names):
			#Read in files, and remove the first column (the first column contains the row numbers)
			angles = pd.read_csv(file, delim_whitespace=True, header=self.header)
			angles = angles.drop(angles.columns[[0]], axis = 1)

			anglesPhi = angles.drop(angles.columns[list(range(len(angles.columns) + 1))[1::2]], axis = 1)
			anglesPsi = angles.drop(angles.columns[list(range(len(angles.columns)))[0::2]], axis = 1)

			ref = pd.read_csv(ref, delim_whitespace=True, header=self.header)
			ref = ref.drop(ref.columns[[0]], axis=1)[:1]

			refPhi = ref.drop(ref.columns[list(range(len(ref.columns) + 1))[1::2]], axis = 1)
			refPsi = ref.drop(ref.columns[list(range(len(ref.columns)))[0::2]], axis = 1)

			refPhi = pd.concat([refPhi] * anglesPhi.shape[0])
			refPsi = pd.concat([refPsi] * anglesPsi.shape[0])

			#Calculate RMSF values all at once
			pd.DataFrame(np.vectorize(self.angleBetween)(anglesPhi, refPhi)).apply(lambda x: x/360).apply(np.square).mean(axis=0).apply(np.sqrt).to_csv(name.replace(self.flag, "Phi"), header=False)
			pd.DataFrame(np.vectorize(self.angleBetween)(anglesPsi, refPsi)).apply(lambda x: x/360).apply(np.square).mean(axis=0).apply(np.sqrt).to_csv(name.replace(self.flag, "Psi"), header=False)

class Wraparound(Algorithm):
	def __init__(self, files: list[str], references: list[str], names: list[str], header: bool) -> None:
		"""
		Args:
			files (list[str]): Files containing the dihedral angle data, columns are expected to be in Phi:x, Psi:x, Phi:x+1, Psi:x+1, etc. form
			references (list[str]): Files containing the reference structures to use in calculating RMSD
			names (list[str]): File names for each RMSD dataset
			header (bool): Do your files have a header?
		"""
		super().__init__(files, references, names, header)

	def RMSD(self, restrictToDBD: bool = False):
		"""
		Calculates the Wraparound RMSD values of a set of dihedral angles, and writes them to a file

		Args:
			restrictToDBD (bool, optional): Should the RMSD be of the whole protein, or DBD only? Defaults to False.
		"""
		for file, ref, name in zip(self.files, self.references, self.names):
			#Read in files, and remove the first column (the first column contains the row numbers)
			angles = pd.read_csv(file, delim_whitespace=True, header=self.header)
			angles = angles.drop(angles.columns[[0]], axis = 1)

			ref = pd.read_csv(ref, delim_whitespace=True, header=self.header)
			ref = ref.drop(ref.columns[[0]], axis=1)[:1]

			if restrictToDBD:
				#Restricts the range to only the DBD (134-312)
				angles = angles.drop(angles.columns[2*311:], axis = 1)
				angles = angles.drop(angles.columns[:132*2], axis = 1)

				ref = ref.drop(ref.columns[2*311:], axis = 1)
				ref = ref.drop(ref.columns[:132*2], axis = 1)
			
			#Calculate RMSD values
			temp = len(angles.columns)
			for x in range(len(angles.columns)//2):
				ref[str(x)] = ref.iloc[:, [2*x, 2*x+1]].apply(lambda x: [x[0], x[1]], axis=1)
				angles[str(x)] = angles.iloc[:, [2*x, 2*x+1]].apply(lambda x: [x[0], x[1]], axis=1)
			ref = ref.drop(ref.columns[:temp], axis=1)
			angles = angles.drop(angles.columns[:temp], axis=1)
			ref = pd.concat([ref] * angles.shape[0])

			pd.DataFrame(np.vectorize(self.calcDistWraparound)(ref, angles)).apply(np.square).mean(axis=1).apply(np.sqrt).to_csv(name, header=False)

	def PairwiseRMSD(self, restrictToDBD: bool = False):
		"""Uses the dihedral angles provided to generate a pairwise matrix of RMSD values, used for K-Means clustering

		Args:
			restrictToDBD (bool, optional): Should only the DBD be considered in the calculation? Defaults to False.
		"""

		for file, ref, name in zip(self.files, self.references, self.names):
			fileList = []
			referenceList = []

			#If file is a list of file names, then read them all in.
			#Otherwise it's only a single file, so read it in and add it to an array 
			if type(file) == list:
				for temp in self.files:
					temp = pd.read_csv(temp, delim_whitespace=True, header=self.header)
					temp = temp.drop(temp.columns[[0]], axis=1)
					fileList.append(temp)
			else:
				temp = pd.read_csv(file, delim_whitespace=True, header=self.header)
				temp = file.drop(temp.columns[[0]], axis=1)
				fileList.append(temp)

			#If file is a list of file names, then read them all in.
			#Otherwise it's only a single file, so read it in and add it to an array 
			if type(self.references[0]) == list:
				for temp in self.references:
					temp = pd.read_csv(temp, delim_whitespace=True, header=self.header)
					temp = temp.drop(temp.columns[[0]], axis=1)
					referenceList.append(temp)
			else:
				temp = pd.read_csv(ref, delim_whitespace=True, header=self.header)
				temp = temp.drop(temp.columns[[0]], axis=1)
				referenceList.append(temp)

			#Since we know that fileList and referenceList are lists (possibly of length one), we can safely do this
			angles = pd.concat(fileList)
			ref = pd.concat(referenceList)

			#Reset the index because when concatenated, each DataFrame brings it's own indexing
			angles = angles.reset_index()
			ref = ref.reset_index()

			if restrictToDBD:
				#Restricts the range to only the DBD (134-312)
				angles = angles.drop(angles.columns[2*311:], axis = 1)
				angles = angles.drop(angles.columns[:132*2], axis = 1)

				ref = ref.drop(ref.columns[2*311:], axis = 1)
				ref = ref.drop(ref.columns[:132*2], axis = 1)

			matrix = []

			num = angles.shape[0]
			for x in range(num):
				currRef = pd.concat([ref.loc[[x]]] * angles.shape[0])
				
				#Calculates the RMSD given the current reference Phi, Psi row

				temp = len(angles.columns)
				for x in range(len(angles.columns)//2):
					currRef[str(x)] = currRef.iloc[:, [2*x, 2*x+1]].apply(lambda x: [x[0], x[1]], axis=1)
					angles[str(x)] = angles.iloc[:, [2*x, 2*x+1]].apply(lambda x: [x[0], x[1]], axis=1)
				currRef = currRef.drop(currRef.columns[:temp], axis=1)
				angles = angles.drop(angles.columns[:temp], axis=1)
				currRef = pd.concat([currRef] * angles.shape[0])

				matrix.append(pd.DataFrame(np.vectorize(self.calcDistWraparound)(currRef, angles)).apply(np.square).mean(axis=1).apply(np.sqrt).to_csv(name, header=False))

				# matrix.append(pd.concat([pd.DataFrame(np.vectorize(self.angleBetween)(currPhi, anglesPhi)), pd.DataFrame(np.vectorize(self.angleBetween)(currPsi, anglesPsi))], axis=1).apply(lambda x: x/360).apply(np.square).mean(axis=1).apply(np.sqrt))

				if x % 100 == 0:
					print(f"{x}/{num} Completed")

			matrix = pd.concat(matrix, axis=1)

			#The pairwise matrix has been generated, but we need to convert it into a format that Amber can read
			with open(name, "w") as csvfile:
				csvwriter = csv.writer(csvfile, delimiter="\t")
				csvwriter.writerow(["#F1", "F2", "RMSD"])
				for col in range(matrix.shape[0]):
					for row, val in enumerate(matrix[col]):
						if row < col + 1:
							continue
						csvwriter.writerow([col+1, row+1, val])

	def RMSF(self):
		"""
		Calculates the Wraparound RMSF values of a set of dihedral angles, and writes them to a file
		"""

		for file, ref, name in zip(self.files, self.references, self.names):
			#Read in files, and remove the first column (the first column contains the row numbers)
			angles = pd.read_csv(file, delim_whitespace=True, header=self.header)
			angles = angles.drop(angles.columns[[0]], axis = 1)

			ref = pd.read_csv(ref, delim_whitespace=True, header=self.header)
			ref = ref.drop(ref.columns[[0]], axis=1)[:1]

			#Calculate RMSF values
			temp = len(angles.columns)
			for x in range(len(angles.columns)//2):
				ref[str(x)] = ref.iloc[:, [2*x, 2*x+1]].apply(lambda x: [x[0], x[1]], axis=1)
				angles[str(x)] = angles.iloc[:, [2*x, 2*x+1]].apply(lambda x: [x[0], x[1]], axis=1)
			ref = ref.drop(ref.columns[:temp], axis=1)
			angles = angles.drop(angles.columns[:temp], axis=1)
			ref = pd.concat([ref] * angles.shape[0])

			pd.DataFrame(np.vectorize(self.calcDistWraparound)(ref, angles)).apply(np.square).mean(axis=0).apply(np.sqrt).to_csv(name, header=False)