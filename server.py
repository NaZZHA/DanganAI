import simul
from helper import *
from flask import *
from forms import *
import os
import random
import json

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = '1c797507e22de341db7c9a8f89df28e5'
app.config['UPLOAD_FOLDER'] = './icons/'
evidence_structure = []
evidences = []
students = []
roster = simul.roster()
messages = []
blackend = 'Random'
trial = simul.class_trial(roster)
phase = 0

@app.route('/init_weights')
def init_weights():
	for i in roster.students:
		i.init_weights(len(roster.students))
	return redirect('/')

@app.route('/load_save', methods=['POST'])
def load_save():
	print(request.files['save_data'])
	f = request.files['save_data']
	fd = f.read().decode("utf-8") 
	json_str = json.loads(fd)
	roster.students.clear()
	evidence_structure.clear()
	for i in json_str['players']:
		s = simul.student(i['name'])
		f = json.dumps(i)
		roster.add_student(s)
		roster.search(i['name']).load(f)
	
	for i in json_str['evidences']:
		evidence_structure.append([i['incriminated'], i['mentioned'], i['content'], i['name']])

	return redirect('/')

@app.route('/save')
def save():
	save_dict = {
		'players' : [],
		'evidences' : []
	}
	for i in roster.students:
		i.save_weights()
		save_dict['players'].append(i.save())

	for i in evidence_structure:
		e = {'name' : i[3], 'incriminated' : i[0], 'mentioned': i[1], 'content': i[2]}
		save_dict['evidences'].append(e)

	f_content = json.dumps(save_dict)
	with open('save.json', 'w') as file:
		json.dump(save_dict, file)
	return redirect('/')


@app.route('/')
def index():
	student_form = student_creation_form()
	if messages:
		return render_template('index.html', form=student_form, students=roster.students, messages=messages[-1])
	else:
		return render_template('index.html', form=student_form, students=roster.students)

@app.route('/set_blackend', methods=['POST'])
def set_blackend():
	name = request.form.get('blackend')
	if name == 'Random':
		s = random.choice(roster.students)
		name = s.name
	blackend = name
	messages.append(f'blackend set to {name}')
	return redirect('/')

@app.route('/wrap_up')
def finale():
	f = trial.final_reward()
	return render_template('reward.html', win=f, students=roster.students_according_to_fitness, blackend=trial.blackend, couples=roster.couples)

@app.route('/evidence_fields')
def evidence():
	return render_template('evidence.html' ,students=roster.students, current_evidences=evidence_structure)

@app.route('/trial_init')
def init_trial():
	print(len(roster.students))
	trial.reset()
	trial.set_blackened(roster.search(blackend))
	for i in roster.students:
		i.fitness = 0
	for i in evidence_structure:
		e = simul.evidence(i[3], roster, i[0], i[1])
		e.content = i[2]
		trial.add_evidence(e)

	for i in trial.evidences:
		i.update()

	return redirect('/main_event')

@app.route('/main_event')
def class_trial():
	events = trial.next_phase_from_server()
	print(events)
	if type(events) is int:
		return redirect('/')
	trial.current_phase += 1
	print(trial.current_phase)
	if trial.current_phase < 4:
		return render_template('trial.html', events=events, search=roster.search)
	else:
		return redirect('/wrap_up')


@app.route('/add_evidence', methods=['POST'])
def add_evidence():
	incriminated = []
	mentioned = []
	content = request.form.get('evidence_content')
	name = request.form.get('evidence_name')



	for i in roster.students:
		is_incriminated = request.form.get(f'{i.name}_incriminated')
		is_mentioned = request.form.get(f'{i.name}_mentioned')
			
		if is_incriminated:
			incriminated.append(i.name)

		elif is_mentioned:
			mentioned.append(i.name)

	evidence_structure.append([incriminated, mentioned, content, name])
	t = [incriminated, mentioned, content, name]

	e = simul.evidence(name, roster, t[0], t[1])
	return redirect('/evidence_fields')

@app.route('/remove_evidence/<evidence_name>')
def remove_evidence(evidence_name):
	for i in evidence_structure:
		if i[3] == evidence_name:
			evidence_structure.remove(i)
	return redirect('/evidence_fields')


@app.route('/icons/<path:filename>')  
def icons(filename):  
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/final_roll_call')
def rollcall():
	message = None
	sl = len(roster.students)
	el = len(evidence_structure)

	print(sl)
	if sl < 3:
		message = 'Not enough students'

	elif el < 2:
		message = 'Not enough evidence'

	if message:
		messages.append(message)
		return redirect('/')

	return render_template('roster.html', students=roster.students)

@app.route('/student_registraition', methods=['POST'])
def student_registraition():
	trust = request.form.get('trust_rating')
	analytical = request.form.get('analytical_rating')
	emotion = request.form.get('emotional_rating')

	if trust == '' or analytical == '' or emotion == '':
		return redirect('/')

	name = request.form.get('name')
	trust = int(trust)
	analytical = int(analytical)
	emotion = int(emotion)

	student = simul.student(name)

	if not os.path.exists('./icons'):
		os.mkdir('./icons')

	if len(request.files)==1:
		file = request.files['icon']
		if file.filename == '':
			roster.add_student(student)
			roster.search(name).init_stats(analytical, trust, emotion)
			return redirect('/')			
		
		x = file.save(os.path.join('./icons', clean_filename(file.filename)))
		student.icon = clean_filename(file.filename)

	student.init_stats(analytical, trust, emotion)
	roster.add_student(student)
	print(student.analytical)
	return redirect('/')

@app.route('/reset_students')
def reset():
	for x in os.listdir('./icons'):
		os.remove(f'./icons/{x}')
	roster.students.clear()
	evidence_structure.clear()
	return redirect('/')

@app.route('/delete_student')
def delete_student():
	if 'student_name' in request.args:
		for i in roster.students:
			if i.name == request.args['student_name']:
				roster.remove_student(request.args['student_name'])
				break
	return redirect('/')

@app.context_processor
def override_url_for():
	return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
	if endpoint == 'static':
		filename = values.get('filename', None)
		if filename:
			file_path = os.path.join(app.root_path,endpoint, filename)
			values['q'] = int(os.stat(file_path).st_mtime)
	return url_for(endpoint, **values)

if __name__ == '__main__':
	app.run(debug=True, port=8000)
