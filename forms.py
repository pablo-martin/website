from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, \
TextAreaField
from wtforms.fields.html5 import DecimalRangeField
from wtforms.validators import ValidationError, DataRequired, Email, EqualTo, \
    Length


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class PostForm(FlaskForm):
    noChars = TextAreaField('# of characters (optional)',
                            validators=[Length(min=1, max=3)])
    temp = TextAreaField('Temperature (optional)')
    post = TextAreaField('Type in phrase to be completed',
                            validators=[Length(min=1, max=100)])
    submit = SubmitField('Autocomplete')
    generate = SubmitField('Generate New Phrase')
    #temperature = DecimalRangeField('temperature', default=0)
