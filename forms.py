from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField
from wtforms.validators import DataRequired

class student_creation_form(FlaskForm):
	name = StringField('name', validators=[DataRequired()])
	trust_rating = IntegerField('trust', validators=[DataRequired()])
	analytical_rating = IntegerField('analytical', validators=[DataRequired()])
	emotional_rating = IntegerField('emotional', validators=[DataRequired()])
	submit = SubmitField('Register', validators=[DataRequired()])
#class evidence_creation_form(FlaskForm):
