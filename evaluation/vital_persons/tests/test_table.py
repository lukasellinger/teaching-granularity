from evaluation.vital_persons.table import project_prediction, evaluate_control, summarize


def row(prediction=None,status='parsed',gold=None):
    return {'case_id':'Q1:birth', 'qid':'Q1', 'parse_status':status,
            'prediction':prediction or {'level':'DAY','year':1900,'month':5,'day':2},
            'gold':gold or {'level':'DAY','year':1900,'month':6,'day':3},
            'target_status':'known','target_level':'YEAR','target_category':'YEAR'}


def test_wrong_day_becomes_correct_year_without_gold_in_projection():
    r=row();before={k:v for k,v in r.items() if k not in {'gold','target_status','target_level','target_category'}}
    projected,status=project_prediction(before,'temporal','cap-YEAR')
    assert projected.year==1900 and status=='parsed'
    result=evaluate_control([r],'temporal','cap-YEAR')
    assert result['truth_accuracy']==1 and result['uniform_error_utility']>0
    assert result['macro_target_match_accuracy']==1


def test_caps_do_not_refine_or_repair_failures():
    r=row(prediction={'level':'CENTURY','century':20})
    projected,_=project_prediction(r,'temporal','cap-YEAR')
    assert projected.level.value=='CENTURY'
    r=row(status='unparsed')
    assert evaluate_control([r],'temporal','cap-YEAR')['uniform_error_utility']==-1
    assert evaluate_control([r],'temporal','always-idk')['uniform_error_utility']==0


def test_location_ancestor_is_from_prediction_not_gold():
    r=row(prediction={'level':'CITY','city':'London','country':'Canada','continent':'North America'})
    projected,_=project_prediction(r,'location','cap-COUNTRY')
    assert projected.country=='Canada' and projected.city is None


def test_macro_weights_categories_equally_and_includes_idk():
    rows=[{'target_category':'YEAR','target_match':True,'truth_correct':True,'utility_positive_informativeness':1,'utility_uniform_wrong_penalty':1,'parse_status':'parsed'} for _ in range(9)]
    rows.append({'target_category':'IDK','target_match':False,'truth_correct':False,'utility_positive_informativeness':0,'utility_uniform_wrong_penalty':-1,'parse_status':'parsed'})
    assert summarize(rows)['macro_target_match_accuracy']==.5



def test_paper_table_layout_includes_accuracy_and_fixed_cap():
    from evaluation.vital_persons.table import render, METHODS, LEVELS
    data = {model: {domain: {
        'methods': {method: {'truth_accuracy': .1*(i+1), 'uniform_error_utility': .05*i}
                    for i, method in enumerate(METHODS)},
        'best_fixed_cap_oracle': {'policy': 'cap-YEAR' if domain == 'temporal' else 'cap-COUNTRY',
                                  'truth_accuracy': .8, 'uniform_error_utility': .2}}
        for domain in LEVELS} for model in ('lfm', 'ministral')}
    _,tex=render(data,'test')
    assert r'\begin{tabular}{lrrrr}' in tex
    assert r'\cmidrule(lr){2-3}' in tex
    assert 'Macro Target Match' not in tex and 'Accuracy' in tex
    assert 'Utility' in tex
    assert 'One-level backoff' not in tex
    assert r'\quad + Best fixed cap' in tex
    assert 'Ministral (base)' in tex and r'\quad FT (ours)' in tex
    assert 'not per question' in tex
