from pydantic import BaseModel
from typing import Literal

class model_input(BaseModel):
    """
    Pydantic class to make handling requests simpler. 

    Pydantic essentially offers a framework to facilitate modeling an
    object in Python with extensive value and type checking capabilities.
    """
    current_status: Literal['Laboratory-confirmed case', 
                            'Probable Case']
    sex:Literal['Male','Female','Unknown','Missing'] = 'Missing'
    age_group: Literal['0 - 9 Years','10 - 19 Years']
    race_ethnicity_combined: Literal['Black, Non-Hispanic',
                                     'American Indian/Alaska Native, Non-Hispanic',
                                     'Asian, Non-Hispanic',
                                     'Hispanic/Latino',
                                     'Multiple/Other, Non-Hispanic',
                                     'Native Hawaiian/Other Pacific Islander, Non-Hispanic',
                                     'White, Non-Hispanic',
                                     'Unknown',
                                     'Missing'] = 'Missing'
    hosp_yn: Literal['Yes','No','Missing','Unknown'] = 'Missing'
    icu_yn: Literal['Yes','No','Missing','Unknown'] = 'Missing'
    medcond_yn: Literal['Yes','No','Missing','Unknown'] = 'Missing'
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        return setattr(self, key, value)
