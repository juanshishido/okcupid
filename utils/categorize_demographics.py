"""
Helper functions for reducing the number of levels in the demographic
columms. These groupings *are* arbitrary.
"""

def religion_categories(religion):
    if len(religion.split()) == 1:
        return 'neutral'
    elif len(religion.split()) > 1:
        return ' '.join(religion.split()[1:])

def job_categories(job):
    if job in ['other', 'rather not say']:
        return 'not specified'
    elif job in ['student', 'education / academia']:
        return 'education'
    elif job in ['science / tech / engineering',
                 'computer / hardware / software']:
        return 'technology'
    elif job in ['artistic / musical / writer', 'entertainment / media']:
        return 'creative'
    elif job in ['sales / marketing / biz dev',
                 'executive / management',
                 'banking / financial / real estate']:
        return 'business'
    elif job in ['law / legal services',
                 'political / government',
                 'transportation',
                 'military']:
        return 'public service'
    elif job in ['unemployed', 'retired']:
        return 'none'
    else:
        return job

def drug_categories(drug):
    if drug in ['sometimes', 'often']:
        return 'yes'
    elif drug == '':
        return 'unknown'
    elif drug == 'never':
        return 'no'
    else:
        return drug

def diet_categories(diet):
    if diet == '':
        return 'unknown'
    elif diet in ['anything', 'mostly anything']:
        return 'no restrictions'
    else:
        return 'restrictions'

def body_categories(body):
    if body in ['fit', 'athletic', 'jacked']:
        return 'in shape'
    elif body in ['', 'rather not say']:
        return 'unknown'
    elif body in ['thin', 'curvy', 'a little extra', 'skinny',
                  'full figured', 'overweight', 'used up']:
        return 'not in shape'
    else:
        return body

def drink_categories(drink):
    if drink in ['very often', 'desperately']:
        return 'very often'
    elif drink == '':
        return 'unknown'
    else:
        return drink
