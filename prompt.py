aspect_description = {
    'persuasiveness': 'This explanation is convincing to me.',
    'transparency' : 'Based on this explanation, I understand why this movie is recommended.',
    'accuracy' : 'This explanation is consistent with my interests.',
    'satisfactory' : 'I am satisfied with this explanation.'
}

LIKERT_INSTRUCT = f'''Try to immerse yourself in this user's experience and provide feedback. 
Consider how the user would feel about the explanatation if he/she were recommended the movie.
Your insights are invaluable. Thank you!'''

# LIKERT_INSTRUCT = f'''
# Assess the 4 asepcts on a 5-point Likert scale, where 1 indicates strongly disagree and 5 indicates strongly agree. 
# Your insights are invaluable. Thank you!'''

def generate_user_profile(row):
    return f'''
        Your gender is : {row['gender']}. 
        Your interest in movies is {row['user_movie_interest']}.
        Number of movies you watch per year is : {row['peryear']}.
        The total number of films you've watched is around : {row['total']}
    '''
    
def prompt_user_profile(user_profile):
    if user_profile['contain'] == True:
        return f'''You are a user of this type:
    {user_profile['prompt']}'''
    else:
        return ''

def prompt_cases(cases):
    if cases['type'] == 'None':
        return ''
    elif 'Personalized' in cases['type']:
        return f'''Previous cases of the user's ratings as reference:
    {cases['prompt']}'''
    else:
        return f'''Previous cases of ratings as reference:
    {cases['prompt']}'''

def prompt_evaluate_in_likert(data):
    print(data)
    return f'''
    Put yourself in the shoes of a user, where a recommendation system has suggested the movie '{data['movie_title']}' to you, accompanied by the following explanation text:

    '{data['explanation']}'
    
    {LIKERT_INSTRUCT}
    
    Persuasiveness: {aspect_description['persuasiveness']}
    Transparency: {aspect_description['persuasiveness']}
    Accuracy: {aspect_description['accuracy']}
    Satisfactory: {aspect_description['satisfactory']}

    Please provide your ratings by entering four numbers separated by spaces.
    Make sure you summarise the scores you have given at the end of your answer which should end in, for example, 3 3 3 3
    '''
    
def prompt_evaluate_in_likert_personalize(data,user_profile):
    print(data)
    return f'''
    Considering you are a user of a movie recommendation platform.
    
    {user_profile}
    
    The recommendation system has suggested a movie to you, accompanied by an explanation text.

     {LIKERT_INSTRUCT}

    Persuasiveness: {aspect_description['persuasiveness']}
    Transparency: {aspect_description['persuasiveness']}
    Accuracy: {aspect_description['accuracy']}
    Satisfactory: {aspect_description['satisfactory']}

    Make sure you summarise the scores you have given at the end of your answer which should end in, for example, 3 3 3 3
    '''
    
def prompt_evaluate_in_likert_personalize_few_shot(data,user_profile,cases):
    return f'''
    Considering you are a user of a movie recommendation platform.
    
    The recommendation system has suggested a movie to you, accompanied by an explanation text.
    Please rate the user experience with the explanation in the following four aspects:

    Persuasiveness: {aspect_description['persuasiveness']}
    Transparency: {aspect_description['persuasiveness']}
    Accuracy: {aspect_description['accuracy']}
    Satisfactory: {aspect_description['satisfactory']}

    Assess the 4 asepcts with integers between 1-5 , where 1 indicates strongly disagree and 5 indicates strongly agree. 
    Please give it, no N/A.

    {prompt_user_profile(user_profile)}

    {prompt_cases(cases)}

    {LIKERT_INSTRUCT}
    
    Movie:'{data['movie_title']}'
    Explanation: '{data['explanation']}'

    Make sure you summarise the scores you have given at the end of your answer. For example , Summary: 3 3 3 3
    '''
    # , Summary: 3 3 3 3
    
def prompt_evaluate_in_likert_personalize_few_shot_per_aspect(data,user_profile,cases,aspect):
    return f'''
    Considering you are a user of a movie recommendation platform.
    
    The recommendation system has suggested a movie to you, accompanied by an explanation text.
    Please rate the user experience with the explanation in the following four aspects:

    {aspect.capitalize()}: {aspect_description[aspect]}

    Assess the asepct with integers between 1-5 , where 1 indicates strongly disagree and 5 indicates strongly agree. 

    {prompt_user_profile(user_profile)}

    {prompt_cases(cases)}

    {LIKERT_INSTRUCT}
    
    Movie:'{data['movie_title']}'
    Explanation: '{data['explanation']}'

    Make sure you summarise the scores you have given at the end of your answer. For example, 3.
    '''

# def prompt_evaluate_in_likert_personalize_one_shot(data,user_profile,case):
#     return f'''
#     Considering you are a user of a movie recommendation platform.
    
#     {prompt_user_profile(user_profile)}
    
#     The recommendation system has suggested the following movies '{data['movie_title']}' to you, accompanied by the following explanation text:

#     '{data['explanation']}'

#     {LIKERT_INSTRUCT}

#     Persuasiveness: {aspect_description['persuasiveness']}
#     Transparency: {aspect_description['persuasiveness']}
#     Accuracy: {aspect_description['accuracy']}
#     Satisfactory: {aspect_description['satisfactory']}
    
#     Previous cases of the user's ratings:
#     {case}


#     Make sure you summarise the scores you have given at the end of your answer which should end in, for example, 3 3 3 3
#     '''


def prompt_evaluate_in_likert_aspect(data,metric):
    print(data)
    return f'''
    Considering you are a user of a movie recommendation platform.
    
    The recommendation system has suggested the movie '{data['movie_title']}' to you, accompanied by the following explanation text:

    '{data['explanation']}'

    {LIKERT_INSTRUCT}

    Persuasiveness: {aspect_description['persuasiveness']}
    Transparency: {aspect_description['persuasiveness']}
    Accuracy: {aspect_description['accuracy']}
    Satisfactory: {aspect_description['satisfactory']}

    Please provide your ratings by entering four integer numbers separated by spaces.
    Make sure you summarise the scores you have given at the end of your answer which should end in, for example, 3 3 3 3
    '''