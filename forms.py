from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    name = StringField('Nombre', validators=[DataRequired()])
    team = SelectField('Equipo', choices=[('team1', 'Team 1'), ('team2', 'Team 2'), ('team3', 'Team 3')], validators=[DataRequired()])
    career = SelectField('Carrera', choices=[('carrera1', 'Carrera 1'), ('carrera2', 'Carrera 2'), ('carrera3', 'Carrera 3')], validators=[DataRequired()])
    submit = SubmitField('Registrar')