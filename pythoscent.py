
# coding: utf-8

# Functions:
#                                                                    # rather have them resources (dico,model pickles)
# - file2ia (your_file)
#         + copy/paste output to dictionary variable (e.g., gui_n = {})
# - ia2d3(gui_n)
# - ia2test_set(gui_n)
#         + copy/paste output to list variable (e.g., test_set_n)    #add commas after each tupple
#                                                                    #text processing issues with score_fam functions
# 
# - test2df (gui_n, test_set_n)                                      #text processing issues here too
#         + give output to variable name (e.g., data_gui_n)
# - plots (data_frame)
# - anovas (data_frame)
# - regressions (data_frame)
# - goal_matrix (data_frame)
# - taskBYmodel_viz(gui, test_set)                                   # var['ls_l'][0]
# - modelsBYtask_viz(gui,'goal', 'target feature')
# - modelsBYset_viz(gui, test_set_gui)                               # var[3]
# - MC_experiment(gui_n, 'made_up_goal', tok_w2v, 'w2v', 10000)
# 

# In[1]:

import pickle, operator, json, requests

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Natural Language Processing modules
from gensim import corpora, models, similarities, utils
from pattern.en import tag, parse
from nltk.corpus import wordnet as wn, stopwords
from nltk.tokenize import RegexpTokenizer

# Data/Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Graph

# Statistics and Machine-learning modules
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn import datasets, linear_model

get_ipython().magic(u'pylab inline')


# # IA handling

# In[ ]:

def file2ia (your_file):
    
    def getLineInfo(fname):

        raw_line = next(fname, None)
        if raw_line is None:
            return False
        full_line = raw_line.rstrip()
        text_only = full_line.lstrip()
        num_tabs = len(full_line) - len(text_only)
        return text_only, num_tabs

    def buildGUI(response, fname):
        # Assume it is a proper tuple:
        key, depth = response
        # Dictionary for this depth:
        local_gui = {}
        # This changes if depth goes up OR down, or reach EOF

        # technically infinite loop, but something will eventually break or return
        while 1:
            # 1. next_depth = same: make previous key/value
            # 2. next_depth +1: call function and make previous key/dict
            # 3. next_depth smaller: make previous key/value
            # 4. EOF: make previous key/value
            next_response = getLineInfo(fname)

            if not next_response:  #end of file
                local_gui[key] = key
                return local_gui, next_response
                break

            # Have a good response at this point
            next_key, next_depth = next_response

            # 2. Set key to result of recursive call
            if next_depth == depth + 1:
                # RECURSIVE CALL
                local_gui[key], last_response = buildGUI(next_response, fname)
                # Last response is next line after nested (recursive) part is finished
                # If depth less than current, return whatever GUI we have
                if last_response:
                    last_key, last_depth = last_response
                    if last_depth < depth:
                        return local_gui, last_response
                    # Otherwise set current key, depth and keep going
                    key, depth = last_response
                else:
                    return local_gui, last_response # RECURSIVE call reached end of file

            # 3. Next line is higher up the IA, so return what we have with next line
            elif next_depth < depth:
                local_gui[key] = key
                return local_gui, next_response
            # 1. same depth, just keep going
            else:
                local_gui[key] = key
                key = next_key

        return local_gui, next_response #print previous_key #, next_key

    my_file = open(your_file, 'r')
    first_response = getLineInfo(my_file)
    print first_response

    # throwaway variable _ will always be False
    GUI, _ = buildGUI(first_response, my_file)

    return GUI

# checking
#file2ia ('../resources/gui_5')


# In[2]:

gui5 = {"car":{ "air conditioning":{ "ventilation settings": "ventilation settings","pulse": "pulse", "continuous": "continuous"}, 
                                    "filter settings":{ "recycle interior air": "recycle interior air", "charcoal mode": "charcoal mode", "pollen mode": "pollen mode"}, 
                                    "temperature settings":{ "display current temperature": "display current temperature", "hotter": "hotter", "colder": "colder"}}, 
                "driving assistance":{ "cruise control":{ "activate": "activate", "turn off": "turn off"}, 
                                        "anti theft protection":{ "choose notification recipient": "choose notification recipient", 
                                        "phone pairing tracking": "phone pairing tracking", "stop vehicle": "stop vehicle"},
                                        "lane change alert":{ "vibrate steering wheel": "vibrate steering wheel", "vibrate pedal": "vibrate pedal", "vibrate rear view mirrors": "vibrate rear view mirrors"},
                                        "gps":{ "check current coordinates": "check current coordinates", "enter destination": "enter destination", "recent destination": "recent destination"}},
                "entertainment":{ "gaming":{ "poker": "poker", "chess": "chess", "online apps": "online apps"},
                                 "television":{ "documentaries": "documentaries", "movies sorted":{ "genre": "genre", "rating": "rating", "release date": "release date"}, 
                                                                "tv series": "tv series"},
                                                                "radio":{ "classic": "classic", "pop": "pop", "electronic": "electronic"}},
                        "phone":{ "contact lists":{ "family": "family", "work": "work", "emergency": "emergency"}, "dial": "dial", "voice mail":{ "listen messages": "listen messages", 
                                                                    "erase last message": "erase last message", "change greetings": "change greetings"}, "pay bills": "pay bills"}}


# In[ ]:

# This function generates IA description and syntax for drawing D3 dendrograms

def ia2d3 (dico):
    
    print 'gui = { "name": "home", "children":['
    
    def nested2d3(dico, indent = '   '):
        for k in dico.keys():
            print indent + '{ "name": "' + str(k) + '",'
            if isinstance(dico[k], dict):
                print indent + '"children":['
                nested2d3(dico[k], indent + "   ")
            if isinstance(dico[k], str):
                print indent + ' "size": 20},'
        print ']},'
    
    nested2d3(dico, indent = '   ')

# Checking    
#ia2d3(gui5)


# # Test set generation

# In[95]:

def wraper1 (gui):
    
    def list_gui_target_features(gui, f_list=[], g_list=[], t_list=[]):
        for k in gui.keys():
            if type(gui[k]) is str:
                f_list.append(gui[k])
                g_list.append('')
                t_list.append('')
            else:
                list_gui_target_features(gui[k])                           
        return zip (f_list, g_list, t_list)
    
    return list_gui_target_features (gui)


# In[96]:

def wraper2 (gui, glob): 
    
    def steps_to_targ_feat (gui, glob , c_list = []):   
        for v in gui.values():   
            if type(v) is dict:
                steps_to_targ_feat(v, glob + 1)
            else:
                c_list.append(glob)     
        return c_list

    return steps_to_targ_feat(gui, glob)


# In[ ]:

def ia2test_set(gui):
    f1 = wraper1 (gui)
    f2 = wraper2 (gui, 1)
    f3 = []
    for i, elem in enumerate(f1):
        temp = []
        for n in elem:
            temp.append(n)
        temp.append(f2[i])
        f3.append(tuple(temp))
    return f3
    
# checking
#test_set_inter = ia2test_set(gui5)
#for t in test_set_inter:
#    print t


# In[3]:

test_set_gui = [('pay bills', 'pay monthly phone plan', 'basic', 2),
('family', 'display address my parents', 'basic', 3),
('change greetings', 'record message welcoming message', 'technical', 3),
('rating', 'list best movies', 'technical', 4)]


# # Wordnet Technicality check

# In[12]:

def score_fam_wn (test_set):
    from nltk.corpus import wordnet as wn, stopwords
    from nltk.tokenize import RegexpTokenizer
    wn_fam = []
    for test in test_set:
        tokenizer = RegexpTokenizer(r'\w+')
        toketxt = tokenizer.tokenize(test[1])
        s = set(stopwords.words('english'))
        filt_wrd = [w for w in toketxt if not w in s]
        w_count = len(filt_wrd)
        g_polys = 0

        for word in filt_wrd:
            w_polys = len(wn.synsets(word))
            g_polys += w_polys

        av_polys = float(g_polys)/float(w_count)
        wn_fam.append(round(av_polys, 2))

    #print sorted(wn_fam), '   >> wordnet familiarity' 
    return wn_fam
    
def score_fam_wn_NN (test_set):

    wn_fam_NN = []

    for test in test_set: 
        filt_wrd = []
        g = test[1]
        for w, pos in tag(g): 
            if pos == "NN":
                filt_wrd.append(w)
        #print filt_wrd

        w_count = len(filt_wrd)
        g_polys = 0

        for word in filt_wrd:
            w_polys = len(wn.synsets(word))
            g_polys += w_polys

        av_polys = float(g_polys)/float(w_count)
        wn_fam_NN.append(round(av_polys, 2))

    #print sorted(wn_fam_NN), '   >> wordnet familiarity for NN pos only'
    return wn_fam_NN

score_fam_wn (test_set_gui)
score_fam_wn_NN (test_set_gui)


# # Load models

# In[13]:

# Load tf-idf representation and dictionary mapping for tokenized corpus
wiki = corpora.MmCorpus('../resources/latent_tok_tfidf.mm')
mon_dico = corpora.Dictionary.load_from_text('../resources/latent_tok_wordids.txt')

# Lemmatized tf-idf representation and dictionary
wiki_lem = corpora.MmCorpus('../resources/latent_lem_tfidf.mm')
mon_dico_lem = corpora.Dictionary.load_from_text('../resources/latent_lem_wordids.txt')

#print "Tokenized corpus:", wiki
#print mon_dico, '\n'
#print "Lemmatized corpus:", wiki_lem
#print mon_dico_lem


# In[14]:

# Load model pickles
tok_w2v = models.word2vec.Word2Vec.load("../resources/word2vec_tok.model")
lem_w2v = models.word2vec.Word2Vec.load("../resources/word2vec_lem.model")
tok_lsi = pickle.load (open ('../resources/pickle_lsi.p', 'rb'))
tok_lda = pickle.load (open ('../resources/pickle_lda.p', 'rb'))
lem_lsi = pickle.load (open ('../resources/pickle_lsi_lem.p', 'rb'))
lem_lda = pickle.load (open ('../resources/pickle_lda_lem.p', 'rb'))


# In[7]:

# Load dico pickles
dico_tok = pickle.load(open('../resources/pickle_dicotok.p', 'rb'))
dico_lem = pickle.load(open('../resources/pickle_dicolem.p', 'rb'))
w2id_l = pickle.load(open('../resources/pickle_w2idlem.p', 'rb'))
w2id_t = pickle.load(open('../resources/pickle_w2idtok.p', 'rb'))


# # Sniffer

# In[15]:

def filter_goal (model, goal, model_type, goal_v = []):
    filter_text_wt = lambda text: [word for word in text if word in w2id_t]
    filter_text_wl = lambda text: [word for word in text if word in w2id_l]
    filter_text = lambda text: [word for word in text if word in dico_tok] 
    filter_text_l = lambda text: [word for word in text if word in dico_lem]
    
    # if Word2Vec models
    if (model_type == "w2v") or (model_type == "w2v_lem"):    
        split_goal = goal.lower().split()
        goal_v = filter_text_wt (split_goal)

        if model_type == "w2v_lem":
            lemm_goal = utils.lemmatize(goal)
            goal_v = filter_text_wl (lemm_goal)
    
    # if latent models
    elif (model_type == "latent") or (model_type == "latent_lem"):
        split_goal = goal.lower().split()
        filter_goal = filter_text(split_goal)
        goal_bow = mon_dico.doc2bow(filter_goal)
        
        if model_type == "latent_lem":
            filter_goal = filter_text_l(utils.lemmatize(goal)) #postag the goal
            goal_bow = mon_dico_lem.doc2bow(filter_goal)
        goal_v = model[goal_bow]
    
    return goal_v     

# Checking
#print filter_goal (lem_w2v, 'call your parents', 'w2v_lem')
#print filter_goal (tok_w2v, 'call your parents', 'w2v')
#print filter_goal (lem_lsi, 'call your parents', 'latent_lem')
#print filter_goal (lem_lda, 'call your parents', 'latent_lem')


# In[16]:

def filter_state (state, model, model_type):
    filter_text_wt = lambda text: [word for word in text if word in w2id_t]
    filter_text_wl = lambda text: [word for word in text if word in w2id_l]
    filter_text = lambda text: [word for word in text if word in dico_tok] 
    filter_text_l = lambda text: [word for word in text if word in dico_lem]
    
    # if Word2Vec models
    if (model_type == "w2v") or (model_type == "w2v_lem"):
        filtered_labels = []
        for label in state.keys():
            
            split_label = label.split()
            filtered_label = filter_text_wt (split_label)
            #filtered_labels.append(filtered_label)
            
            if model_type == "w2v_lem":
                split_label = utils.lemmatize(label)
                filtered_label = filter_text_wl (split_label)
            filtered_labels.append(filtered_label)
        
        return filtered_labels
    
    
    elif (model_type == "latent") or (model_type == "latent_lem"):
        ar = []
        for label in state.keys():
            split_label = label.split()
            filter_label = filter_text(split_label)
            
            if model_type == "latent_lem":
                filter_label = filter_text_l(utils.lemmatize(label))
            ar.append(filter_label)
        
        state2index = [mon_dico.doc2bow(txt) for txt in ar]
        if model_type == "latent_lem":
            state2index = [mon_dico_lem.doc2bow(txt) for txt in ar]
        
        return state2index        

# Checking    
#filter_state (gui5, tok_w2v, 'w2v')
#filter_state (gui5, lem_lsi,'latent_lem')


# In[21]:

def get_sorted_similarity (state, model, goal, model_type):
    sorted_sims = []
    similaritiz = {}
    
    if (model_type == "w2v") or (model_type == "w2v_lem"):
        filtered_goal = filter_goal(model, goal, model_type)
        filtered_state = filter_state (state, model, model_type)
        for i, fs in enumerate(filtered_state):
            sim = model.n_similarity(fs, filtered_goal)
            similaritiz[state.keys()[i]] = np.around(sim, decimals=3)   #adding the similarity scores for any label
            # Check in case similarity score is an array
            if str(type(np.around(sim, decimals=3))) == "<type 'numpy.ndarray'>":
                similaritiz[state.keys()[i]] = 0
        
    # LSI/LDA models
    elif (model_type == "latent") or (model_type == "latent_lem"):
        filtered_goal = filter_goal(model, goal, model_type)
        state2index = filter_state (state, model, model_type)

        index = similarities.MatrixSimilarity(model[state2index], num_features=29000)
        indexed_scores = list(index[filtered_goal])
        for index, score in enumerate(indexed_scores):
            label = state.keys()[index]
            similaritiz[label] = np.around(score, decimals=3) # round(score,3)
            # Check in case similarity score is an array
            if str(type(np.around(score, decimals=3))) == "<type 'numpy.ndarray'>":
                similaritiz[label] = 0

    # Return label/scores dict sorted by values
    sorted_sims = sorted(similaritiz.items(), key=operator.itemgetter(1), reverse=True) 
    return sorted_sims                              #returns a list of tuples i.e [(,), (,)]

# Checking
#print get_sorted_similarity(gui5, lem_w2v, 'call your parents', 'w2v_lem')
#print get_sorted_similarity(gui5, tok_w2v, 'call your parents', 'w2v')
#print get_sorted_similarity(gui5, lem_lsi, 'call your parents', 'latent_lem')
#print get_sorted_similarity(gui5, tok_lsi, 'call your parents', 'latent'), "\n"

#print get_sorted_similarity(gui5, tok_w2v, 'call your parents as soon as you reach the shopping center', 'w2v')
#print get_sorted_similarity(gui5, lem_lsi, 'call your parents as soon as you reach the shopping center', 'latent_lem')
#print get_sorted_similarity(gui5, tok_lsi, 'call your parents as soon as you reach the shopping center', 'latent')


# In[ ]:

def sniffer_steps(gui, model, goal, target_feat, model_type, global_count = 0):
    sorted_state = get_sorted_similarity(gui, model, goal, model_type)
    found_goal = False
    #print sorted_state
    for label, score in sorted_state:
        #print label, score
        global_count += 1
        if type(gui[label]) is dict:
            found_goal, g_cnt = sniffer_steps(gui[label], model, goal, target_feat, model_type)
            global_count += g_cnt
            if found_goal:
                return True, global_count
    
        elif gui[label] == target_feat:
            return True, global_count
    
    return False, global_count

# Checking
#print sniffer_steps(gui5, tok_w2v, 'check how much I scored last week at chess tournament', 'chess', 'w2v'), '\n'
#print sniffer_steps(gui5, tok_w2v, 'control my speed on the highway', 'activate', 'w2v'), '\n'
#print sniffer_steps(gui5, tok_w2v, 'let me drive full speed', 'turn off', 'w2v'), '\n'

#print sniffer_steps(gui5, lem_w2v, 'check how much I scored last week at chess tournament', 'chess', 'w2v_lem'), '\n'
#print sniffer_steps(gui5, lem_w2v, 'control my speed on the highway', 'activate', 'w2v_lem'), '\n'
#print sniffer_steps(gui5, lem_w2v, 'let me drive full speed', 'turn off', 'w2v_lem'), '\n'


#print sniffer_steps(gui5, lem_w2v, 'disable overtaking alarm', 'vibrate pedal', 'w2v_lem'), '\n'
#print sniffer_steps(gui5, lem_lsi, 'disable overtaking alarm', 'vibrate pedal', 'latent_lem'), '\n'
#print sniffer_steps(gui5, tok_lda, 'set the alarm that I am being overtaken to be on the pedal', 'vibrate pedal', 'latent'), '\n'


# # Model tester

# In[ ]:

def model_test(gui, test_set, model, model_type):
    
    wn_fam = score_fam_wn (test_set_gui)
    wn_fam_NN = score_fam_wn_NN (test_set_gui)
    
    test_metrics =[]
    goal_l = []
    target_l=[]
    prescrit=[]
    fam = []
    
    for a, b, c, d in test_set:
        goal_l.append(b)
        target_l.append(a)
        fam.append(c)
        prescrit.append(d)
        result = sniffer_steps(gui, model, b, a, model_type)
        test_metrics.append(result[1])
    
    #print test_metrics
    return zip(target_l, goal_l, test_metrics, prescrit, fam, wn_fam, wn_fam_NN)

# checking
#model_test(gui5, test_set_gui, tok_w2v, 'w2v')


# # Dataframe

# In[ ]:

def test2df (gui_n, test_set_n):
    tok_w2v_test = model_test(gui_n, test_set_n, tok_w2v, 'w2v')
    lem_w2v_test = model_test(gui_n, test_set_n, lem_w2v, 'w2v_lem')
    tok_lsi_test = model_test(gui_n, test_set_n, tok_lsi, 'latent')
    tok_lda_test = model_test(gui_n, test_set_n, tok_lda, 'latent')
    lem_lsi_test = model_test(gui_n, test_set_n, lem_lsi, 'latent_lem')
    lem_lda_test = model_test(gui_n, test_set_n, lem_lda, 'latent_lem')

    df_tok_w2v = pd.DataFrame(tok_w2v_test)
    df_tok_w2v['model']='word2vec'
    df_tok_w2v['prepping']='tokens'

    df_lem_w2v = pd.DataFrame(lem_w2v_test)
    df_lem_w2v['model']='word2vec'
    df_lem_w2v['prepping']='lemmas'

    df_tok_lsi = pd.DataFrame(tok_lsi_test)
    df_tok_lsi['model']='lsi'
    df_tok_lsi['prepping']='tokens'

    df_tok_lda = pd.DataFrame(tok_lda_test)
    df_tok_lda['model']='lda'
    df_tok_lda['prepping']='tokens'

    df_lem_lsi = pd.DataFrame(lem_lsi_test)
    df_lem_lsi['model']='lsi'
    df_lem_lsi['prepping']='lemmas'

    df_lem_lda = pd.DataFrame(lem_lda_test)
    df_lem_lda['model']='lda'
    df_lem_lda['prepping']='lemmas'

    frames = [df_tok_w2v, df_lem_w2v, df_tok_lsi, df_tok_lda, df_lem_lsi, df_lem_lda]
    df_gui_n = pd.concat(frames)
    df_gui_n.columns = ['target_feat','goal',  'counts','prescrite','familiarity','wn_familiarity','wn_familiarity_NN','model', 'prepping']

    return df_gui_n

#checking
#
#a = test2df (gui5, test_set_gui)
#a.head (5)


# In[109]:




# # Results

# In[110]:

def plots (data_frame):
    #boxplots
    data_frame.boxplot(column = 'counts', by = 'model')
    data_frame.boxplot(column = 'counts', by = 'prepping')
    data_frame.boxplot(column = 'counts', by = 'familiarity')
    data_frame.boxplot(column = 'counts', by = 'prescrite')
    
    #factor plots
    e = sns.factorplot(x="familiarity", y="counts", col = 'model', hue="prepping",data=data_frame,
                       palette="YlGnBu_d", size=5, aspect=.75)
    e.despine(left=True)

def regressions (data_frame):
    selecta = data_frame[data_frame['model'] != 'lda']
    #print 'Regression familiarity vs wn_NN'

    X = selecta['wn_familiarity_NN']
    Y = selecta['counts']

    regr = linear_model.LinearRegression()
    regr.fit (X[:,np.newaxis], Y)

    plt.xlabel('familiarity')
    plt.scatter(X, Y, color='red')
    plt.plot(X, regr.predict(X[:,np.newaxis]), color='blue')
    plt.show()

    # The coefficients
    #print 'Coefficients: %.3f' % regr.coef_[0]
    # The mean square error
    #print "Mean residual sum of squares: %.1f" % np.mean((regr.predict(X[:,np.newaxis]) - Y) ** 2)
    # Explained variance score: 1 is perfect prediction
    #print('Variance of Y explained by X: %.3f' % regr.score(X[:,np.newaxis], Y))
    
    #print 'Regression familiarity vs wn'
    X = selecta['wn_familiarity']
    Y = selecta['counts']

    regr = linear_model.LinearRegression()
    regr.fit (X[:,np.newaxis], Y)

    plt.xlabel('familiarity')
    plt.scatter(X, Y, color='red')
    plt.plot(X, regr.predict(X[:,np.newaxis]), color='blue')
    plt.show()

    # The coefficients
    #print 'Coefficients: %.3f' % regr.coef_[0]
    # The mean square error
    #print "Mean residual sum of squares: %.1f" % np.mean((regr.predict(X[:,np.newaxis]) - Y) ** 2)
    # Explained variance score: 1 is perfect prediction
    #print('Variance of Y explained by X: %.3f' % regr.score(X[:,np.newaxis], Y))
    
def anovas (data_frame):
    
    anova_result = ols('counts ~ C(model, Sum)*C(prepping, Sum)*C(familiarity, Sum)',
               data=data_frame).fit()
    table = sm.stats.anova_lm(anova_result, typ=2) # Type 2 ANOVA DataFrame
    #print table


# In[ ]:

# checking
#plots (a)


# In[ ]:

#anovas (a)


# In[ ]:

#regressions (a)


# # Visualizations

# In[ ]:

# this is to see how sensible is your test_set and how much it has been trimmed by the filtering functions
# Only done for w2v tokenized

def goal_matrix (test_set):
    filtered_set = list(test_set)
    filter_text = lambda text: [word for word in text.split() if word in w2id_t]
    
    for i,line in enumerate(filtered_set):
        f = filter_text(line[0])
        g = filter_text(line[1])
        filtered_set[i]= (' '.join(f), ' '.join(g), line[2], line[3])
    print filtered_set
    
    goal_index = []
    for goal in filtered_set:
        goal_index.append(goal[1])

    sim_heatmap = pd.DataFrame(index=goal_index)

    for i, feat in enumerate(filtered_set):
        sim_features = []
        for j, goal in enumerate(filtered_set):
            sim_features.append(tok_w2v.n_similarity(feat[0].split(), goal[1].split()) )
        sim_heatmap[feat[0]] = sim_features
    sim_heatmap.head()

    # Plot similarity matrix as heatmap
    f, ax = plt.subplots(figsize=(6, 7))

    # Draw the heatmap 
    sns.heatmap(sim_heatmap,cmap="YlGnBu",
                robust=True,#annot=True,
                square=True, xticklabels=True, yticklabels=True,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
#checking
#goal_matrix (test_set_gui)

# In[124]:

def sniffer_wrapper (gui, model, goal, target_feat, model_type):    
    
    def sniffer_logfile(gui, model, goal, target_feat, model_type, global_count = 0, seq=[]):
        state = get_sorted_similarity(gui, model, goal, model_type)
        found_goal = False

        for label, score in state:
            #print '"%s", %s, ' %(label, round(score, 2))
            seq.append(label)    #seq is the sequence of steps navigated by sniffer
            global_count += 1

            if type(gui[label]) is dict:
                found_goal, g_cnt, seq = sniffer_logfile(gui[label], model, goal, target_feat, model_type)
                global_count += g_cnt

                if found_goal:
                    return True, global_count, seq
                seq.append(label)

            elif gui[label] == target_feat:
                return True, global_count, seq

        return False, global_count, seq   
    
    return sniffer_logfile(gui, model, goal, target_feat, model_type)

#sniffer_wrapper(gui, tok_w2v,'invite friend play chess', 'chess', 'w2v')

def graphviz_wrapper (gui, model, goal, t_feat, mod_type):    

    # Instantiating graphviz object
    dig = Graph('task', engine = 'fdp')
    dig.body.append('size="10"')
    dig.node_attr.update(color='floralwhite', style='filled')
    dig.graph_attr.update(splines='curved')
    
    # Specifying nodes and edges from sniffer seq(uence) output
    def nodes_edges (gui, model, goal, t_feat, mod_type):
        log = []
        result = []
        result = sniffer_wrapper (gui, model, goal, t_feat, mod_type)
        log = result[2]
        log.insert(0, 'START')
        log.append('END')

        decallage = log[1:]
        logfile = zip(decallage,log)
        for line in logfile:
            #print ( 'dig.edge("%s", "%s")' % (line[0], line[1]))
            if model == tok_w2v:
                dig.edge(line[0], line[1], color="green")
            elif model == lem_w2v:
                dig.edge(line[0], line[1], color="blue")
            elif model == tok_lsi:
                dig.edge(line[0], line[1], color="pink")
            elif model == lem_lsi:
                dig.edge(line[0], line[1], color="violet")
    
    nodes_edges (gui, model, goal, t_feat, mod_type)
    return dig             

#graphviz_wrapper(gui, tok_w2v,'invite friend play chess', 'chess', 'w2v')


# In[127]:

# this is to graphviz any given task of your test set for any chosen model

def taskBYmodel_viz(gui, test_set):
    som = dict()
    w2_t =[]
    w2_l =[]
    ls_t =[]
    ld_t =[]
    ls_l =[]
    ld_l =[]
    for t_feat, goal, bla, blah in test_set:
        w2_t.append (graphviz_wrapper(gui, tok_w2v, goal, t_feat, 'w2v'))
        w2_l.append (graphviz_wrapper(gui, lem_w2v, goal, t_feat, 'w2v_lem'))
        ls_t.append (graphviz_wrapper(gui, tok_lsi, goal, t_feat, 'latent'))
        ld_t.append (graphviz_wrapper(gui, tok_lda, goal, t_feat, 'latent'))
        ls_l.append (graphviz_wrapper(gui, lem_lsi, goal, t_feat, 'latent_lem'))
        ld_l.append (graphviz_wrapper(gui, lem_lda, goal, t_feat, 'latent_lem'))
    som['w2_t'] = w2_t
    som['w2_l'] = w2_l
    som['ls_t'] = ls_t
    som['ld_t'] = ld_t
    som['ls_l'] = ls_l
    som['ld_l'] = ld_l
    
    print 'Assign output to variable. \n var["model_choice"][goal number] \n Model choices: w2_t, w2_l, ls_t, ls_l, ld_t, ld_l. '
    return som

# Checking
#t = taskBYmodel_viz(gui5, test_set_gui)


# In[ ]:

#t['w2_t'][1]


# In[ ]:

# this is to graphviz any made-up task (goal description + target feature) for all 4 trained models
# trained models are LSA and w2v, lemmatized and tokenized

def modelsBYtask_viz (gui, goal, t_feat):    

    # Instantiating graphviz object
    dig = Graph('task', engine = 'fdp')
    dig.body.append('size="10"')
    dig.node_attr.update(color='floralwhite', style='filled')
    dig.graph_attr.update(splines='curved')
    
    # Specifying nodes and edges from sniffer seq(uence) output
    def nodes_edges (gui, goal, t_feat):
        log = []
        result = []
        models = [(tok_w2v, 'w2v'),(tok_lsi,'latent'),(tok_lda, 'latent'),(lem_w2v, 'w2v_lem'), 
                  (lem_lsi, 'latent_lem'), (lem_lda, 'latent_lem')]
        for model, mod_type in models: 
        
            result = sniffer_wrapper (gui, model, goal, t_feat, mod_type)
            log = result[2]
            log.insert(0, 'START')
            log.append('END')

            decallage = log[1:]
            logfile = zip(decallage,log)
            for line in logfile:
                #print ( 'dig.edge("%s", "%s")' % (line[0], line[1]))
                if model == tok_w2v:
                    dig.edge(line[0], line[1], color="lightblue")
                elif model == lem_w2v:
                    dig.edge(line[0], line[1], color="blue")
                elif model == tok_lsi:
                    dig.edge(line[0], line[1], color="pink")
                elif model == lem_lsi:
                    dig.edge(line[0], line[1], color="violet")
    
    #print 'Pinkishes are LSA, Blueishes are w2v, lemmatized is brighter'
    nodes_edges (gui, goal, t_feat)
    return dig             

# Checking
#modelsBYtask_viz (gui5, 'do your daily chess training', 'chess')


# In[119]:

# this is to graphviz your test set for all 4 trained models
def modelsBYset_viz (gui, test_set):
    my_dig = None
    dig_list = []
    for t_feat, goal, c, d in test_set:
        dig_list.append  (modelsBYtask_viz(gui, goal, t_feat))
    return dig_list 


# In[ ]:

h = modelsBYset_viz (gui5, test_set_gui)
h[0]


# # Basic MC simulation using w2v
def MC_experiment(gui, goal, model, model_type, N):

    def get_weights (state, goal, model, model_type, weights={}):

        # Filter goal and features in current state
        filtered_goal = filter_goal(model, goal, model_type)
        filtered_state = filter_state (state, model, model_type)

        # Store the labels
        label_list = []

        if (model_type == "w2v") or (model_type == "w2v_lem"):
            for i, label in enumerate(filtered_state):
                label_list.append(model.n_similarity(label, filtered_goal))
                if type(state[state.keys()[i]]) is dict:
                    get_weights (state[state.keys()[i]], goal, model, model_type, weights)

        # Use absolute values for cosine similarities
        #print type(label_list), len(label_list)
        arr = np.array(label_list)
        val_abs = abs(arr)
        # Normalize so weights --> probabilities
        reweighted = val_abs.transpose()/np.sum(val_abs, axis=0).transpose()

        # Zip up labels w/ corresponding weights
        for label, proba in zip (state.keys(), reweighted):
            weights[label] = proba  

        return weights

    goal_weights = get_weights(gui, goal, model, model_type)

    def simulate_1_pass_down (state, w, counts = {}):
        #w = get_weights (state, goal, model, model_type,  weights={})
        labels = state.keys()
        poids = [w[label] for label in labels]

        # Winner will be an array of length 1, so...
        probs = np.array(poids)
        probs /= probs.sum()
        winner = np.random.choice(labels, 1, p = probs)
        # Convert to string
        winner = winner[0]

        if winner in counts:
            counts[winner] += 1
        else:
            counts[winner] = 1
        if type(state[winner]) is dict:
            simulate_1_pass_down (state[winner], w)

        return counts

    output = ""
    for i in range(N):
        output = simulate_1_pass_down (gui, goal_weights)
        
    return output
# In[17]:

def MC_experiment(gui, goal, model, model_type, N):

    def get_weights (state, goal, model, model_type, weights={}):

        # Filter goal and features in current state
        filtered_goal = filter_goal(model, goal, model_type)
        filtered_state = filter_state (state, model, model_type)

        # Store the labels
        label_list = np.array([])

        if (model_type == "w2v") or (model_type == "w2v_lem"):
            for i, label in enumerate(filtered_state):
                #label_list.append(model.n_similarity(label, filtered_goal))
                label_list = np.append(label_list, np.array(model.n_similarity(label, filtered_goal)))
                if type(state[state.keys()[i]]) is dict:
                    get_weights (state[state.keys()[i]], goal, model, model_type, weights)

        # Use absolute values for cosine similarities
        arr = np.array(label_list)
        val_abs = np.absolute(arr)
        
        # Normalize so weights --> probabilities
        reweighted = val_abs.transpose()/np.sum(val_abs, axis=0).transpose()

        # Zip up labels w/ corresponding weights
        for label, proba in zip (state.keys(), reweighted):
            weights[label] = proba  

        return weights

    goal_weights = get_weights(gui, goal, model, model_type)

    def simulate_1_pass_down (state, w, counts = {}):
        #w = get_weights (state, goal, model, model_type,  weights={})
        labels = state.keys()
        poids = [w[label] for label in labels]

        # Winner will be an array of length 1, so...
        probs = np.array(poids)
        probs /= probs.sum()
        winner = np.random.choice(labels, 1, p = probs)
        # Convert to string
        winner = winner[0]

        if winner in counts:
            counts[winner] += 1
        else:
            counts[winner] = 1
        if type(state[winner]) is dict:
            simulate_1_pass_down (state[winner], w)

        return counts

    output = ""
    for i in range(N):
        output = simulate_1_pass_down (gui, goal_weights)
        
    return output


# In[18]:

#mc_results = MC_experiment(gui5, 'dial', tok_w2v, 'w2v', 10000)


# In[19]:

#sorted_mc_rez = sorted(mc_results.items(), key=operator.itemgetter(1), reverse = True)
#for s in sorted_mc_rez:
#    print s


# In[ ]:



