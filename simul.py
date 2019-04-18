import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from statistics import median
import os


students = []
save_directory = './character_data'
sess = tf.Session()

if not os.path.exists(save_directory):
	os.mkdir(save_directory)

class student:
	def __init__(self, name):
		self.blackend = False
		self.name = name
		self.survived = False
		self.fitness = 0
		self.intellect = 0
		self.last_event = np.array([[0,0,0,0]], dtype=np.float32)
		self.weights = []
		self.path = save_directory + f'/{self.name}/'

		students.append(self)
		
	def init_stats(self, analytical, trust, emotional):
		self.intellect = tf.Constant(intellect)
		self.luck = tf.Constant(luck)
		self.trust = tf.Constant(trust)

	def save_weights(self):
		if not os.path.exists(self.path):
			os.mkdir(self.path)

		for i in self.weights:
			x = sess.run(i)
			np.save(self.path + f'/{self.weights.index(i)}', x)

	def load_weights(self):
		files = []
		for i in range(len(self.weights)):
			files.append(f'{i}.npy')

		loaded_weights = []
		for i in files:
			loaded_weights.append(np.load(self.path + i))

		for x, y in zip(self.weights, loaded_weights):
			ind = self.weights.index(x)
			self.weights[ind] = tf.Variable(y.tolist())

		self.set_weights()

	def init_weights(self):
		####################
		# CLASS TRIAL VARS #
		####################
		self.shapes = []
		self.w1_evidence = tf.Variable(tf.random_normal([len(students), 7]))
		self.w2_evidence = tf.Variable(tf.random_normal([7, 2]))
		self.output_evidence = tf.Variable(tf.random_normal([2, 1]))	

		self.w1_event = tf.Variable(tf.random_normal([4, 2]))
		self.w2_event = tf.Variable(tf.random_normal([2, 2]))
		self.output_event = tf.Variable(tf.random_normal([2, 4]))	

		self.w1_final = tf.Variable(tf.random_normal([4, 2]))
		self.w2_final = tf.Variable(tf.random_normal([2, 1]))
		self.output_final = tf.Variable(tf.random_normal([1, 2]))

		self.w1_accuse = tf.Variable(tf.random_normal([4, 3]))
		self.w2_accuse = tf.Variable(tf.random_normal([3, 2]))
		self.output_accuse = tf.Variable(tf.random_normal([2, len(students)]))

		self.weights += [self.w1_evidence, self.w2_evidence, self.output_evidence, self.w1_event, self.w2_event, self.output_event, self.w1_final, self.w2_final, self.output_final, self.w1_accuse, self.w2_accuse, self.output_accuse]
		sess.run(tf.global_variables_initializer())
		for i in self.weights:
			x = sess.run(i)
			y = x.shape
			self.shapes.append(y)

	def set_weights(self):
		self.w1_evidence = self.weights[0]
		self.w2_evidence = self.weights[1]
		self.output_evidence = self.weights[2]
		self.w1_event = self.weights[3]
		self.w2_event = self.weights[4]
		self.output_event = self.weights[5]
		self.w1_final = self.weights[6]
		self.w2_final = self.weights[7]
		self.output_final = self.weights[8]
		self.w1_accuse = self.weights[9]
		self.w2_accuse = self.weights[10]
		self.output_accuse = self.weights[11]
		sess.run(tf.global_variables_initializer())


	def mutate(self, weights):
		p1 = random.randrange(0, len(weights)-1)
		p2 = random.randrange(0, len(weights)-1)

		if p2 > p1:
			temp = p1
			p1 = p2
			p2 = p1

		mutated_weights = weights[:]
		lesion = weights[p1:p2]
		random.shuffle(lesion)
		mutated_weights[p1:p2] = lesion
		return mutated_weights

	def crossver(self, other):
		for x1, y1, shape, ind in zip(self.weights, other.weights, self.shapes, range(len(self.weights))):
			area = shape[0] * shape[1]
			x2 = sess.run(x1)
			y2 = sess.run(y1)

			x2 = x2.reshape([area])
			y2 = y2.reshape([area])

			x2 = x2.tolist()
			y2 = y2.tolist()

			x3 = None
			y3 = None

			for x,y in zip(x2, y2):
				choice = random.choice([True, False])
				if choice:
					x2[x2.index(x)] = y
					y2[y2.index(y)] = x

				x3 = np.array(x2, dtype=np.float32)
				x3 = x3.reshape(shape)

				y3 = np.array(y2, dtype=np.float32)
				y3 = y3.reshape(shape)	

			x2 = self.mutate(x2)
			y2 = self.mutate(y2)

			self.weights[ind] = tf.Variable(x3.tolist())
			other.weights[ind] = tf.Variable(y3.tolist())

			self.set_weights()
			other.set_weights()


	def evidence_analysis(self, evidence):
		inputs = tf.placeholder(tf.float32, shape=(4, len(students)))

		l1 = tf.matmul(inputs, self.w1_evidence)
		l1 = tf.nn.leaky_relu(l1)

		l2 = tf.matmul(l1, self.w2_evidence)
		l2 = tf.nn.leaky_relu(l2)

		output_layer = tf.matmul(l2, self.output_evidence)
		output_layer = tf.nn.sigmoid(output_layer)
		output_layer = tf.reshape(output_layer, [1,4])

		out = sess.run(output_layer, feed_dict={inputs: evidence})

		return out

	def prev_event_analysis(self, prev_event):
		inputs = tf.placeholder(tf.float32, shape=[1, 4])

		l1 = tf.matmul(inputs, self.w1_event)
		l1 = tf.nn.leaky_relu(l1)

		l2 = tf.matmul(l1, self.w2_event)
		l2 = tf.nn.leaky_relu(l2)

		output_layer = tf.matmul(l2, self.output_event)
		output_layer = tf.nn.sigmoid(output_layer)

		out = sess.run(output_layer, feed_dict={inputs: prev_event})

		return out

	def decide(self, evidence, accused):

		a1 = self.evidence_analysis(evidence).tolist()[0]
		a2 = self.prev_event_analysis(self.last_event).tolist()[0]
		#a3 = self.suspect_analysis(accused).tolist()[0]

		inputs = np.array([a1, a2], dtype=np.float32)
		l1 = tf.matmul(inputs, self.w1_final)
		l1 = tf.nn.leaky_relu(l1)

		l2 = tf.matmul(l1, self.w2_final)
		l2 = tf.nn.leaky_relu(l2)

		output_layer = tf.matmul(l2, self.output_final)
		output_layer = tf.nn.softmax(output_layer)
		output_layer = tf.reshape(output_layer, [1, 4])

		out = sess.run(output_layer)
		decision = []
		for i in out[0]:
			if i == max(out[0]):
				decision.append(1)
			else:
				decision.append(0)

		self.last_event = np.array([decision])
		return decision

		#ACTIONS
		# 1. Accuse
		# 2. Agree
		# 3. Deny
		# 4. State

	def accuse(self, evidence):
		inputs = self.evidence_analysis(evidence)
		l1 = tf.matmul(inputs, self.w1_accuse)
		l1 = tf.nn.leaky_relu(l1)

		l2 = tf.matmul(l1, self.w2_accuse)
		l2 = tf.nn.leaky_relu(l2)

		output_layer = tf.matmul(l2, self.output_accuse)
		output_layer = tf.nn.softmax(output_layer)

		out = sess.run(output_layer)
		decision = []
		for i in out[0]:
			if i == max(out[0]):
				decision.append(1)
			else:
				decision.append(0)

		return decision

class roster:
	def __init__(self, starting_roster=[], initialize_variables=True):
		self.students = starting_roster
		self.students_according_to_fitness = self.students

		if initialize_variables:
			for i in self.students:
				i.init_weights()


	def add_student(self, student):
		self.students.append(student)
		for i in self.students:
			i.init_weights()

	def sort_according_to_fitness(self):
		self.students_according_to_fitness.sort(key=lambda x: x.fitness)
		
	def search(self, name):
		for i in self.students:
			if i.name.lower() == name.lower():
				return i
		return None

	def get_median_fitness(self):
		fitness = [i.fitness for i in self.students]
		return median(fitness)

	def save_student_weights(self):
		for i in self.students:
			i.save_weights()

	def load_student_weights(self):
		for i in self.students:
			i.load_weights()

	def __iadd__(self, student):
		self.students.append(student)

class class_trial:
	def __init__(self, roster):
		self.num_of_phases = 3
		self.moves = 0
		self.current_phase = 0
		self.roster = roster
		self.evidences = []
		self.current_evidence = 0
		self.current_suspect = np.zeros([1, len(self.roster.students)])
		self.blackend_votes = 0
		self.blackend_deny = 0
		self.stated_evidences = []
		self.blackend = None
		self.max_moves = 5

		self.rewards = {
			'correct_accusation' : 50,
			'incorrect_accustation' : -20,
			'correct_consent' : 40,
			'incorrect_consent' : -10,
			'statement_of_evidence' : 20,
			'correct_counter' : 45,
			'incorrect_counter' : -10,
			'survivor_win' : 75,
			'blackend_win' : 75,
			'blackend_defeat' : 75
		}

		for i in self.roster.students:
			i.fitness = 0

	def add_evidence(self, *evidences):
		self.evidences += evidences

	def set_blackened(self, person=None):
		blackend = random.choice(self.roster.students)
		if type(person) is student:
			blackend = person
		print(f'{blackend.name} IS THE BLACKEND')
		self.blackend = blackend
		blackend.blackend = True
		return blackend

	def final_reward(self):
		ind = self.current_suspect
		print(self.roster.students[ind.index(1)].name)
		if self.roster.students[ind.index(1)] == self.blackend:
			for i in self.roster.students:
				if not i == self.blackend:
					i.fitness += self.rewards['survivor_win']
				else:
					i.fitness -= 200
		else:
			self.blackend.fitness += self.rewards['blackend_win']

		rost.sort_according_to_fitness()

	def reset(self):
		self.moves = 0
		self.current_phase = 0
		self.current_evidence = 0
		self.blackend_votes = 0
		self.blackend_deny = 0
		self.stated_evidences = []

	def next_phase(self):
		self.current_phase += 1
		print(f'PHASE{self.current_phase}')
		self.max_moves = random.randrange(4, 14)
		for i in range(self.max_moves):
			choosing = random.choice(self.roster.students)

			if self.moves == 0:
				self.current_evidence = random.choice(self.evidences)
				self.current_evidence.state(choosing)
				self.current_evidence_with_statment = self.current_evidence.state(choosing)
				self.current_suspect = choosing.accuse(self.current_evidence_with_statment)
				self.moves += 1
				continue
			self.moves += 1
			action = choosing.decide(self.current_evidence_with_statment, self.current_suspect)
			act = None
			if action[0] == 1:
				act = 'Accuse'
				self.current_suspect = choosing.accuse(self.current_evidence_with_statment)
				if self.roster.students[self.current_suspect.index(1)] == self.blackend:
					choosing.fitness += self.rewards['correct_accusation']
				else:
					choosing.fitness -= self.rewards['incorrect_accustation']
					self.blackend.fitness += 2	

				self.blackend_votes = 0
				self.blackend_deny = 0			

			if action[1] == 1:
				act = 'Agree'
				self.blackend_votes += 1

				if self.blackend_deny >= 3:
					self.blackend_votes = 0
					self.blackend_deny = 0

				if self.blackend_votes >= int(len(self.roster.students) * 0.75):
					self.final_reward()
					return 0

				if self.roster.students[self.current_suspect.index(1)] == self.blackend:
					choosing.fitness += self.rewards['correct_consent']
				else:
					choosing.fitness += self.rewards['incorrect_consent']

			if action[2] == 1:
				act = 'Deny'
				self.blackend_deny += 1
				if self.blackend_deny >= 3:
					self.blackend_votes = 0
					self.blackend_deny = 0

				if not self.roster.students[self.current_suspect.index(1)] == self.blackend:
					choosing.fitness += self.rewards['correct_counter']
				else:
					choosing.fitness += self.rewards['incorrect_counter']

			if action[3] == 1:
				act = 'State'
				self.current_evidence = random.choice(self.evidences)
				self.current_evidence_with_statment = self.current_evidence.state(choosing)
				choosing.fitness += self.rewards['statement_of_evidence']

			print(f'{choosing.name} : {act}, {self.roster.students[self.current_suspect.index(1)].name}')


		if self.current_phase >= self.num_of_phases:
			self.final_reward()
			return 0
		else:
			self.next_phase()

class evidence:
	def __init__(self, roster, incriminated, mentioned):
		self.student_roster = roster
		self.incriminated = self.__get_incriminated(incriminated)
		self.mentioned = self.__get_mentioned(mentioned)
		self.uninvolved = self.__get_uninvolved()
		self.statement = ''

	def __get_incriminated(self, names):
		incriminated = []
		for s in self.student_roster.students:
			if s.name in names:
				incriminated.append(1)
			else:
				incriminated.append(0)
		return incriminated

	def __get_mentioned(self, names):
		mentioned = []
		for s in self.student_roster.students:
			if s.name in names:
				mentioned.append(1)
			else:
				mentioned.append(0)
		return mentioned

	def __get_uninvolved(self):
		uninvolved = []
		for s in self.student_roster.students:
			if self.incriminated[self.student_roster.students.index(s)] == 1 or self.mentioned[self.student_roster.students.index(s)] == 1:
				uninvolved.append(0)
			else:
				uninvolved.append(1)
		return uninvolved	

	def set_statement(self, statement):
		self.statement = statement

	def state(self, speaker):
		speaker_ind = []
		for s in self.student_roster.students:
			if type(speaker) is str:
				if s.name == speaker:
					speaker_ind.append(1)
				else:
					speaker_ind.append(0)

			elif s.name == speaker.name:
				speaker_ind.append(1)
			else:
				speaker_ind.append(0)

		return np.array([self.incriminated, self.mentioned, self.uninvolved, speaker_ind])



names = ['Dazo', 'Kaede', 'Akko', 'Tachanka', 'Cutab', 'Montagne', 'Shuichi', 'Sucy', 'Ochako', 'Lotte']

rost = roster([student(i) for i in names])

dazo = rost.search('dazo')
akko = rost.search('akko')
kaede = rost.search('kaede')

e1 = evidence(rost, ['Dazo', 'Sucy', 'Ochako'], ['Kaede', 'Akko', 'Shuichi', 'Lotte'])
e2 = evidence(rost, ['Tachanka', 'Akko', 'Dazo'], ['Montagne', 'Cutab', 'Shuichi'])
e3 = evidence(rost, ['Dazo'], ['Tachanka', 'Kaede', 'Akko'])
e4 = evidence(rost, ['Montagne', 'Kaede'], ['Akko', 'Sucy', 'Cutab', 'Ochako'])
e5 = evidence(rost, ['Cutab', 'Akko'], ['Sucy'])
e6 = evidence(rost, ['Kaede', 'Shuichi', 'Lotte'], ['Tachanka', 'Montagne'])
e7 = evidence(rost, ['Lotte', 'Akko', 'Sucy'], ['Kaede', 'Shuichi'])
e8 = evidence(rost, ['Dazo'], ['Lotte', 'Akko', 'Sucy'])

rost.load_student_weights()

f1 = e1.state(kaede)
def run(steps=2):
	for i in range(steps):
		print(f'\n###########################\nTRIAL {i+1}\n###########################\n')
		trial = class_trial(rost)
		trial.add_evidence(e1, e2, e3, e4, e5, e6, e7, e8)
		trial.set_blackened(dazo)
		trial.next_phase()	

		for s in rost.students:
			print(f'{s.name} : {s.fitness}')

		students_according_to_fitness = rost.students_according_to_fitness[::-1]
		#students_according_to_fitness.remove(trial.blackend)
		halfpoint = int(len(students_according_to_fitness) / 2) - 1

		for x in students_according_to_fitness[:halfpoint-1]:
			temp = students_according_to_fitness[:halfpoint-1]
			temp.remove(x)
			mate = random.choice(temp)
			x.crossver(mate)
			print(f'genetic crossover between {x.name} and {mate.name}')

		for x in students_according_to_fitness[halfpoint+1:]:
			temp = students_according_to_fitness[halfpoint+1:]
			temp.remove(x)
			mate = random.choice(temp)
			x.crossver(mate)
			print(f'genetic crossover between {x.name} and {mate.name}')

	rost.save_student_weights()

run()
sess.close()