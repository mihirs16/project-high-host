import model_init as mod_i
import model_pickle_files as mod_pic
import corpora_machine as corps

from nltk.stem import PorterStemmer

# model first train
# import model_train

# corpora machine
text_idf = corps.corpora_train()

# model predict and generate tag list
ml_features, ml_features_models = mod_i.load_tag_machines()

def text_return_tags(text, title):

    # clean text
    cleaned_text = corps.clean_text(text)
    cleaned_title = corps.clean_text(title)
    keywd_by_freq = corps.freq_dist(cleaned_text)

    # new text to features
    text_ft = text_idf.transform([cleaned_text])

    # predict tags
    tag_list = []
    
    for model_index in range(0, len(ml_features_models)):
        
        # text features from articles
        y_pred = ml_features_models[model_index].predict(text_ft)
        if y_pred == 1:
            tag_list.append(ml_features[model_index])

        # title
        if ml_features[model_index].lower() in title or ml_features[model_index].lower() in cleaned_title:
            tag_list.append(ml_features[model_index])
        
        # frequency distribution
        if ml_features[model_index].lower() in keywd_by_freq:
            tag_list.append(ml_features[model_index])

    # return tags
    return tag_list

def test_webscraper_function(url):
    import selenium
    import bs4
    from bs4 import BeautifulSoup
    from selenium import webdriver

    # Getting Pages
    driver = webdriver.Chrome('chromedriver.exe')
    driver.get(url)
    res = driver.execute_script("return document.documentElement.outerHTML")
    driver.quit()

    # Parse Page
    soup = BeautifulSoup(res, 'lxml')

    # Text
    para = soup.findAll('p')
    text = ''
    for p in para:
        text = text + ' ' + p.getText()
    # text = text_processor(text)

    try:
            name = soup.find('h1').getText()
    except:
        
        name = 'None'

    return text, name

# local testing
# text, title = test_webscraper_function('https://elemental.medium.com/did-you-have-coronavirus-without-knowing-it-d33bbce9e9e5')
# print(text_return_tags(text, title))