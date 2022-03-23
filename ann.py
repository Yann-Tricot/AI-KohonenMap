# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Réseaux de neurones
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# pour les réseaux de neurones profond: https://playground.tensorflow.org/
# pour les gaz neuronaux and co: https://www.demogng.de/
# ------------------------------------------------------------------------


import numpy as np #librairie de calcul matriciel
import matplotlib.pyplot as plt # librairie d'affichage
import networkx as nx # librairie d'affichage de réseaux


# fonctions de calculs pour l'activité d'un neurone
# NB on aurait pu passer par des classes, mais rester au niveau des fonctions permet une meilleure performance

# fonctions d'aggrégations possibles pour un neurone

"""
@summary: Fonction d'aggrégation qui fait la somme pondérée des entrées
@param connection_weight: le vecteur de poids des connections du neurone
@type connection_weight: numpy.array
@param input_activity: le vecteur des activités d'entrée du neurone
@type input_activity: numpy.array
"""
def aggregator_weighted_sum(connection_weight,input_activity):
    return np.dot(connection_weight,input_activity)



# fonctions d'activation possibles pour un neurone

"""
@summary: Fonction d'activation identité
@param activity: l'activité en sortie de la fonction d'aggrégation
@type activity: float
"""
def activation_function_identity(activity):
    return activity

"""
@summary: Fonction d'activation créneau (-1 si <0, 0 si 0, 1 si >0)
@param activity: l'activité en sortie de la fonction d'aggrégation
@type activity: float
"""
def activation_function_step(activity):
    return np.where(activity>0,1,np.where(activity<0,-1,0))

"""
@summary: Fonction d'activation tangeante hyperbolique
@param activity: l'activité en sortie de la fonction d'aggrégation
@type activity: float
"""
def activation_function_tanh(activity):
    return np.tanh(activity)




# fonctions d'apprentissage possibles pour un neurone

"""
@summary: Fonction d'apprentissage de Hebb
@param connection_weight: le vecteur de poids des connections du neurone
@type connection_weight: numpy.array
@param input_activity: le vecteur des activités d'entrée du neurone
@type input_activity: numpy.array
@param output_activity: la valeur d'activation du neurone
@type output_activity: numpy.array d'un seul élément (voir la classe Neuron)
@param params: les différents paramètres de la règle d'apprentissage: 
'learning_rate': le taux d'apprentissage
@type params: dictionary
"""
def learn_hebb(connection_weight,input_activity,output_activity,params):
    return params['learning_rate']*input_activity*output_activity

"""
@summary: Fonction d'apprentissage de Hebb bornée
@param connection_weight: le vecteur de poids des connections du neurone
@type connection_weight: numpy.array
@param input_activity: le vecteur des activités d'entrée du neurone
@type input_activity: numpy.array
@param output_activity: la valeur d'activation du neurone
@type output_activity: numpy.array d'un seul élément (voir la classe Neuron)
@param params: les différents paramètres de la règle d'apprentissage: 
'learning_rate': le taux d'apprentissage
'vmin': la valeur minimale du poids d'une connection
'vmax': la valeur minimale du poids d'une connection
@type params: dictionary
"""
def learn_hebb_bound(connection_weight,input_activity,output_activity,params):
    return np.minimum(np.maximum(params['learning_rate']*input_activity*output_activity,params['vmin']-connection_weight),params['vmax']-connection_weight)

"""
@summary: Fonction d'apprentissage de Oja
@param connection_weight: le vecteur de poids des connections du neurone
@type connection_weight: numpy.array
@param input_activity: le vecteur des activités d'entrée du neurone
@type input_activity: numpy.array
@param output_activity: la valeur d'activation du neurone
@type output_activity: numpy.array d'un seul élément (voir la classe Neuron)
@param params: les différents paramètres de la règle d'apprentissage: 
'learning_rate': le taux d'apprentissage
@type params: dictionary
"""
def learn_oja(connection_weight,input_activity,output_activity,params):
    return params['learning_rate']*output_activity*(input_activity-connection_weight*output_activity)




class Neuron:
    """ Classe représentant un neurone """

    """
    @summary: Constructeur d'un neurone
    @param aggregator_function: fonction d'aggregation
    @type aggregator_function: fonction qui prend en paramètre le vecteur de poids et le vecteur d'entrées
    @param activation_function: fonction d'activation
    @type activation_function: fonction qui prend en paramètre l'activité (aggrégée)
    @param learn_function: fonction d'apprentissage
    @type learn_function: fonction qui prend en paramètre le vecteur de poids, le vecteur d'activité d'entrée, l'activité du neurone et des paramètres
    @param learn_function_param: les paramètres de la fonction d'apprentissage
    @type learn_function_param: dictionnary
    @param activity: activité initiale du neurone
    @type activity: numpy.array d'un seul élément (permet un passage par référence pour que la mise à jour de l'activité soit visible au niveau du réseau)
    @param connection_weight_mask: masque de connection (i.e. 1 si le neurone est connecté au ième neurone, 0 sinon)
    @type connection_weight_mask: numpy.array (de valeur 0/1)
    @param connection_weight: poids des connections
    @type connection_weight: numpy.array
    """
    def __init__(self, aggregator_function, activation_function, learn_function, learn_function_param, activity, connection_weight_mask, connection_weight):
        self.__aggregator_function = aggregator_function
        self.__activation_function = activation_function
        self.__learn_function = learn_function
        self.__learn_function_param = learn_function_param
        self.__activity = activity
        self.__connection_weight_mask = connection_weight_mask
        self.__connection_weight = connection_weight

    """
    @summary: modifie l'activité du neurone (sert pour les entrées)
    @param activity: la nouvelle activité
    @type activity: float
    """
    def set_activity(self,activity):
        self.__activity[:] = activity

    """
    @summary: calcule l'activité du neurone
    @param input_activity: l'activité d'entrée reçue
    @type input_activity: numpy.array
    """
    def compute(self,input_activity):
        self.__activity[:] = self.__activation_function(self.__aggregator_function(self.__connection_weight,input_activity))

    """
    @summary: modifie les poids des connections (appelle la règle d'apprentissage)
    @param input_activity: l'activité d'entrée reçue
    @type input_activity: numpy.array
    """
    def learn(self,input_activity):
        self.__connection_weight[:] += self.__learn_function(self.__connection_weight,input_activity,self.__activity, self.__learn_function_param)*self.__connection_weight_mask




class ANN:
    """ Classe représentant un réseau de neurones artificiel"""
# NB on pourrait la définir comme un ensemble de neurones, mais on garde ici la main sur les activités et connexions pour faciliter la communication avec la visu + des performances correctes (pour pouvoir manipuler de la matrice)

    """
    @summary: Constructeur d'un réseau de neurone
    @param nb_neurons: le nombre de neurones
    @type nb_neurons: int
    @param aggregator_function: fonction d'aggregation utilisée par les neurones
    @type aggregator_function: fonction qui prend en paramètre le vecteur de poids et le vecteur d'entrées
    @param activation_function: fonction d'activation utilisée par les neurones
    @type activation_function: fonction qui prend en paramètre l'activité (aggrégée)
    @param learn_function: fonction d'apprentissage utilisée par les neurones
    @type learn_function: fonction qui prend en paramètre le vecteur de poids, le vecteur d'activité d'entrée, l'activité du neurone et des paramètres
    @param learn_function_param: les paramètres de la fonction d'apprentissage utilisée par les neurones
    @type learn_function_param: dictionnary
    @param mode_input_activity: la structure du réseau (i.e. quels neurones sont des entrées)
    'all': tous les neurones sont des entrées mais tous vont également calculer leur activité
    'alone': tous les neurones sont des entrées sauf le dernier neurone qui est le seul à calculer son activité
    @type mode_input_activity: string
    @param mode_connection: 'all_other' pour que chaque neurone soit connecté à tous les autres neurones, 'random_other' pour que chaque neurone soit connnecté à un certain pourcentage (param_connection 'percentage_connection') des autres neurones, 'alone' pour que tous les neurones soit connectés au dernier
    @type mode_connection: string
    @param param_connection: paramètres pour le mode 'random_other': 'percentage_connection' le pourcentage de connections aux autres neurones
    @type param_connection: dictionary
    @param mode_weight: random' pour des poids aléatoires (entre param_weight 'vmin' et 'vmax'), 'zeros' pour des poids nuls
    @type mode_weight: string
    @param param_weight: paramètres pour le mode 'random': 'vmin' la valeur minimale d'un poids initial, 'vmax' la valeur maximale d'un poids initial
    @type param_weight: dictionary
    """
    def __init__(self, nb_neurons, aggregator_function, activation_function, learn_function, learn_function_param, mode_input_activity, mode_connection, param_connection, mode_weight, param_weight):
        self.nb_neurons = nb_neurons
        self.__input_mask = self.__create_input_mask(mode_input_activity)
        self.__activity_mask = self.__create_activity_mask(mode_input_activity)
        self.activity = np.random.rand(self.nb_neurons)
        self.connection_weight_mask = self.__create_connection_mask(mode_connection,param_connection)
        self.connection_weight = self.__create_connection(mode_weight,param_weight)
        self.__neurons_input = []
        self.__neurons_compute = []
        for i in range(self.nb_neurons):
            neuron = Neuron(aggregator_function, activation_function, learn_function, learn_function_param,self.activity[i:i+1],self.connection_weight_mask[:,i],self.connection_weight[:,i])
            if self.__activity_mask[i]==1:            
                self.__neurons_compute.insert(i,neuron) #NB même si i est plus grand que la bonne position, ça le met à la fin
            if self.__input_mask[i]==1:            
                self.__neurons_input.insert(i,neuron) #NB idem
            
    """
    @summary: modifie l'activité des neurones d'entrées
    @param activity: le vecteur d'activités
    @type activity: numpy.array
    """
    def set_input(self,input_activity):
        for i in range(len(self.__neurons_input)):
            self.__neurons_input[i].set_activity(input_activity[i])

    """
    @summary: calcule l'activité de l'ensemble des neurones
    @param propagate: faux si chaque neurone calcule une fois son activité, vrai si la mise à jour des activités doit se faire itérativement jusqu'à l'absence de changement
    @type propagate: boolean
    """
    def compute(self,propagate):
        old_activity = self.activity.copy()
        for neuron in self.__neurons_compute:
            neuron.compute(old_activity)
        new_activity = self.activity
        if propagate:
            while (np.sum(np.abs(new_activity-old_activity))!=0):
                old_activity[:] = new_activity
                for neuron in self.__neurons_compute:
                    neuron.compute(old_activity)
                new_activity[:] = self.activity

    """
    @summary: modifie les poids des connections de l'ensemble des neurones
    """
    def learn(self):
        for neuron in self.__neurons_compute:
            neuron.learn(self.activity)
        
    """
    @summary: entraîne le réseau de neurone
    @param params: les paramètres de l'entraînement
    @type params: dictionary
    @param verbose: vrai si les valeurs des poids doivent s'afficher, faux sinon
    @type verbose: boolean
    @param visu: vrai si un affichage graphique du réseau doit être fait, faux sinon
    @type visu: boolean
    """
    def train(self, params, verbose, visu):
        raise NotImplementedError

    """
    @summary: teste le réseau de neurone
    @param dataset: la base de données de test
    @type dataset: dictionary
    """
    def test(self, dataset):
        for i in range(dataset.shape[0]):
            self.set_input(dataset[i])
            self.visu.show_test_input("exemple "+str(i))
            self.compute(True)
            self.visu.show_test_result()

    """
    @summary: attribue un module graphique de visualisation
    @param visu: le module graphique de visualisation
    @type visu: Graph_ANN
    """
    def set_visu(self, visu):
        self.visu = visu

    """
    @summary: Création du masque (1 = vrai, 0 = faux) pour les entrées (i.e. les neurones du modèle qui servent d'entrée et dont l'activité va donc être forcée par l'extérieur)
    @param mode: 'all' pour que tous les neurones soit des entrées, 'alone' pour ne calculer que tous les neurones sauf le dernier soit des entrées (même mode pour le masque des connexions)
    @type mode: string
    """
    def __create_input_mask(self,mode):
        if mode=='all':
            return np.ones((self.nb_neurons,))
        elif mode=='alone':
            input_mask = np.ones((self.nb_neurons,))
            input_mask[self.nb_neurons-1] = 0
            return input_mask

    """
    @summary: Création du masque (1 = vrai, 0 = faux) pour les activités (i.e. les neurones du modèle dont l'activité doit être calculée)
    @param mode: 'all' pour que tous les neurones calculent leur activité, 'alone' pour ne calculer que le dernier neurone (même mode que pour le masque des entrées)
    @type mode: string
    """
    def __create_activity_mask(self,mode):
        if mode=='all':
            return self.__input_mask[:]
        elif mode=='alone':
            return 1-self.__input_mask[:]


    """
    @summary: Création du masque (1 = vrai, 0 = faux) pour les connections (i.e. les neurones du modèle qui sont connectés entre eux, indépendemment de la valeur du poids)
    @param mode: 'all_other' pour que chaque neurone soit connecté à tous les autres neurones, 'random_other' pour que chaque neurone soit connnecté à un certain pourcentage (params 'percentage_connection') des autres neurones, 'alone' pour que tous les neurones soit connectés au dernier
    @type mode: string
    @param params: paramètres pour le mode 'random_other': 'percentage_connection' le pourcentage de connections aux autres neurones
    @type params: dictionary
    """
    def __create_connection_mask(self,mode,params):
        if mode=='all_other':
            connection_weight_mask = np.ones((self.nb_neurons,self.nb_neurons))
            connection_weight_mask[:] *= 1-np.diag(np.ones((self.nb_neurons,)))
            return connection_weight_mask        
        elif mode=='random_other':
            connection_weight_mask = np.where(np.random.rand(self.nb_neurons,self.nb_neurons)<params['percentage_connection']/100.,1.,0.)
            connection_weight_mask[:] *= 1-np.diag(np.ones((self.nb_neurons,)))
            return connection_weight_mask
        elif mode=='alone':
            connection_weight_mask = np.zeros((self.nb_neurons,self.nb_neurons))
            connection_weight_mask[0:self.nb_neurons-1,self.nb_neurons-1] = np.ones(self.nb_neurons-1,)
            return connection_weight_mask

    """
    @summary: Création de la matrice de poids de connections
    @param mode: 'random' pour des poids aléatoires (entre params 'vmin' et 'vmax'), 'zeros' pour des poids nuls
    @type mode: string
    @param params: paramètres pour le mode 'random': 'vmin' la valeur minimale d'un poids initial, 'vmax' la valeur maximale d'un poids initial
    @type params: dictionary
    """
    def __create_connection(self,mode,params):
        if mode=='random':
            connection_weight = np.random.rand(self.nb_neurons,self.nb_neurons)*(params['vmax']-params['vmin'])+params['vmin']
        elif mode=='zeros':
            connection_weight = np.zeros((self.nb_neurons,self.nb_neurons))
        connection_weight[:] *= self.connection_weight_mask
        return connection_weight




class Graph_ANN:
    """ Classe d'affichage graphique d'un réseau de neurones artificiel"""

    """
    @summary: Constructeur d'un affihage graphique d'un réseau de neurone
    @param ann: le réseau de neurones à afficher
    @type ann: ANN
    """
    def __init__(self, ann):
        self.__ann = ann
        self.__graph = nx.OrderedDiGraph()
        for i in range(self.__ann.nb_neurons):
            self.__graph.add_node(str(i))
        for i in range(self.__ann.nb_neurons):
            for j in range(self.__ann.nb_neurons):
                if self.__ann.connection_weight_mask[i,j]!=0:
                    self.__graph.add_edge(str(i),str(j))
        self.__ann.set_visu(self)

    """
    @summary: affiche le réseau de neurone
    @param title: le titre de la figure
    @type title: string
    """
    def show(self,title=''):
        plt.figure("modèle "+self.__ann.name+" "+title)
        lci = np.where(self.__ann.connection_weight_mask.flatten()!=0)
        nx.draw_shell(self.__graph,width=np.abs(self.__ann.connection_weight.flatten()[lci]),edge_color=np.where(self.__ann.connection_weight.flatten()[lci]>0,'b','r'),node_color=np.where(self.__ann.activity==0,'k',np.where(self.__ann.activity>0,'b','r')),node_size=300+30*np.abs(self.__ann.activity))
        plt.draw()
        plt.show()

    """
    @summary: affiche du réseau de neurone (utilisé pour montrer l'entrée reçue lors d'un test)
    @param title: le titre de la figure
    @type title: string
    """
    def show_test_input(self,title=''):
        plt.figure("test modèle "+self.__ann.name+" "+title)
        plt.subplot(121)
        lci = np.where(self.__ann.connection_weight_mask.flatten()!=0)
        nx.draw_shell(self.__graph,width=np.abs(self.__ann.connection_weight.flatten()[lci]),edge_color=np.where(self.__ann.connection_weight.flatten()[lci]>0,'b','r'),node_color=np.where(self.__ann.activity==0,'k',np.where(self.__ann.activity>0,'b','r')),node_size=300+30*np.abs(self.__ann.activity))
        plt.draw()

    """
    @summary: affiche du réseau de neurone (utilisé pour montrer l'activité résultante à une entrée reçue lors d'un test)
    @param title: le titre de la figure
    @type title: string
    """
    def show_test_result(self):
        plt.subplot(122)
        lci = np.where(self.__ann.connection_weight_mask.flatten()!=0)
        nx.draw_shell(self.__graph,width=np.abs(self.__ann.connection_weight.flatten()[lci]),edge_color=np.where(self.__ann.connection_weight.flatten()[lci]>0,'b','r'),node_color=np.where(self.__ann.activity==0,'k',np.where(self.__ann.activity>0,'b','r')),node_size=300+30*np.abs(self.__ann.activity))
        plt.draw()
        plt.show()
    
    """
    @summary: affiche le réseau de neurone de manière interactive
    """
    def show_update(self):
        plt.clf()
        lci = np.where(self.__ann.connection_weight_mask.flatten()!=0)
        nx.draw_shell(self.__graph,width=np.abs(self.__ann.connection_weight.flatten()[lci]),edge_color=np.where(self.__ann.connection_weight.flatten()[lci]>0,'b','r'),node_color=np.where(self.__ann.activity==0,'k',np.where(self.__ann.activity>0,'b','r')),node_size=300+30*np.abs(self.__ann.activity))
        plt.pause(0.0001)
        plt.draw()



class Hebb_ANN(ANN):
    """ Classe correspondant à un réseau de neurones artificiel utilisant la règle de Hebb"""
    
    """
    @summary: Constructeur du réseau de neurone
    @param nb_neurons: le nombre de neurones
    @type nb_neurons: int
    @param activation_function: fonction d'activation utilisée par les neurones
    @type activation_function: fonction qui prend en paramètre l'activité (aggrégée)
    @param learn_function_param: les paramètres de la fonction d'apprentissage de Hebb 
    'learning_rate': le taux d'apprentissage
    @type learn_function_param: dictionnary
    @param param_connection: les paramètres d'initialisation des connections
    'percentage_connection': le pourcentage de connection par neurone
    @type param_connection: dictionary
    @param param_weight: les paramètres d'initialisation des poids des connections
    'vmin': la valeur minimale d'un poids
    'vmax': la valeur maximale d'un poids 
    @type param_weight: dictionary
    """
    def __init__(self, nb_neurons, activation_function, learn_function_param, param_connection, param_weight):
        ANN.__init__(self, nb_neurons, aggregator_weighted_sum, activation_function, learn_hebb, learn_function_param, 'all', 'random_other', param_connection, 'random', param_weight)
        self.name = 'Hebb'

    """
    @summary: entraîne le réseau de neurone
    @param params: les paramètres de l'entraînement
    'epochs': le nombre de pas d'apprentissage
    @type params: dictionary
    @param verbose: vrai si les valeurs des poids doivent s'afficher, faux sinon
    @type verbose: boolean
    @param visu: vrai si un affichage graphique du réseau doit être fait, faux sinon
    @type visu: boolean
    """
    def train(self, params, verbose, visu):
        if verbose:
            print("poids avant apprentissage\n",self.connection_weight)
        if visu:
            self.visu.show("avant apprentissage")
            plt.ion()
            self.visu.show("pendant apprentissage")
        for i in range(params['epochs']):
            self.compute(False)
            self.learn()
            if visu:
                self.visu.show_update()
        if verbose:
            print("poids après apprentissage\n",self.connection_weight)
        if visu:
            plt.ioff()
            plt.close()
            self.visu.show("après apprentissage")


class Hebb_Bounded_ANN(Hebb_ANN):
    """ Classe correspondant à un réseau de neurones artificiel utilisant la règle de Hebb bornée"""
    # NB hérite de Hebb_ANN pour ne pas avoir à redéfinir le train
    
    """
    @summary: Constructeur du réseau de neurone
    @param nb_neurons: le nombre de neurones
    @type nb_neurons: int
    @param activation_function: fonction d'activation utilisée par les neurones
    @type activation_function: fonction qui prend en paramètre l'activité (aggrégée)
    @param learn_function_param: les paramètres de la fonction d'apprentissage de Hebb 
    'learning_rate': le taux d'apprentissage
    'vmin': la valeur minimale du poids d'une connection
    'vmax': la valeur minimale du poids d'une connection
    @type learn_function_param: dictionnary
    @param param_connection: les paramètres d'initiliasation des connections
    'percentage_connection': le pourcentage de connection par neurone
    @type param_connection: dictionary
    @param param_weight: les paramètres d'initialisation des poids des connections
    'vmin': la valeur minimale d'un poids
    'vmax': la valeur maximale d'un poids 
    @type param_weight: dictionary
    """
    def __init__(self, nb_neurons, activation_function, learn_function_param, param_connection, param_weight):
        ANN.__init__(self, nb_neurons, aggregator_weighted_sum, activation_function, learn_hebb_bound, learn_function_param, 'all', 'random_other', param_connection, 'random', param_weight)
        self.name = 'Hebb'


class Oja_ANN(Hebb_ANN):
    """ Classe correspondant à un réseau de neurones artificiel utilisant la règle de Oja"""
    # NB hérite de Hebb_ANN pour ne pas avoir à redéfinir le train mais aussi parce que la règle Oja est dérivée de celle de Hebb
    
    """
    @summary: Constructeur du réseau de neurone
    @param nb_neurons: le nombre de neurones
    @type nb_neurons: int
    @param activation_function: fonction d'activation utilisée par les neurones
    @type activation_function: fonction qui prend en paramètre l'activité (aggrégée)
    @param learn_function_param: les paramètres de la fonction d'apprentissage de Oja 
    'learning_rate': le taux d'apprentissage
    @type learn_function_param: dictionnary
    @param param_connection: les paramètres d'initiliasation des connections
    'percentage_connection': le pourcentage de connection par neurone
    @type param_connection: dictionary
    @param param_weight: les paramètres d'initialisation des poids des connections
    'vmin': la valeur minimale d'un poids
    'vmax': la valeur maximale d'un poids 
    @type param_weight: dictionary
    """

    def __init__(self, nb_neurons, activation_function, learn_function_param, param_connection, param_weight):
        ANN.__init__(self, nb_neurons, aggregator_weighted_sum, activation_function, learn_oja, learn_function_param, 'all', 'random_other', param_connection, 'random', param_weight)
        self.name = 'Oja'



class Oja_PCA_ANN(Oja_ANN):
    """ Classe correspondant à un réseau de neurones artificiel utilisant la règle de Oja pour la PCA"""
    # NB hérite de Oja_ANN parce que c'est la règle de Oja mais utilisée sur une autre structure de réseau
    
    """
    @summary: Constructeur du réseau de neurone
    @param nb_neurons: le nombre de neurones
    @type nb_neurons: int
    @param learn_function_param: les paramètres de la fonction d'apprentissage de Oja 
    'learning_rate': le taux d'apprentissage
    @type learn_function_param: dictionnary
    @param param_weight: les paramètres d'initialisation des poids des connections
    'vmin': la valeur minimale d'un poids
    'vmax': la valeur maximale d'un poids 
    @type param_weight: dictionary
    """

    def __init__(self, nb_neurons, learn_function_param, param_weight):
        ANN.__init__(self, nb_neurons, aggregator_weighted_sum, activation_function_identity, learn_oja, learn_function_param, 'alone', 'alone', None, 'random', param_weight)
        self.name = 'Oja appliqué à la PCA'

    """
    @summary: entraîne le réseau de neurone
    @param params: les paramètres de l'entraînement
    'epochs': le nombre de pas d'apprentissage
    @type params: dictionary
    @param verbose: vrai si les valeurs des poids doivent s'afficher, faux sinon
    @type verbose: boolean
    @param visu: vrai si un affichage graphique du réseau doit être fait, faux sinon
    @type visu: boolean
    """
    def train(self, params, verbose, visu):
        if verbose:
            print("poids avant apprentissage\n",self.connection_weight)
        if visu:
            self.visu.show("avant apprentissage")
            plt.ion()
            self.visu.show("pendant apprentissage")
        dataset = params['dataset']
        dataset_size = dataset.shape[0]
        for i in range(params['epochs']):
            self.set_input(dataset[i%dataset_size])
            self.compute(False)
            self.learn()
            if visu:
                self.visu.show_update()
        if verbose:
            print("poids après apprentissage\n",self.connection_weight)
        if visu:
            plt.ioff()
            plt.close()
            self.visu.show("après apprentissage")


class Hopfield_ANN(Hebb_ANN):
    """ Classe correspondant à un réseau de neurones artificiel utilisant la règle de Oja pour la PCA"""
    # NB hérite de Hebb_ANN car la règle de Hopfield est dérivée de celle de Hebb
    
    """
    @summary: Constructeur du réseau de neurone
    @param nb_neurons: le nombre de neurones
    @type nb_neurons: int
    @param learn_function_param: les paramètres de la fonction d'apprentissage de Oja 
    'learning_rate': le taux d'apprentissage (forcément égale à 1 / nombre d'exemples dans le dataset d'apprentissage)
    @type learn_function_param: dictionnary
    """
    def __init__(self, nb_neurons, learn_function_param):
        ANN.__init__(self, nb_neurons, aggregator_weighted_sum, activation_function_step, learn_hebb, learn_function_param, 'all', 'all_other', None, 'zeros', None)
        self.name = 'Hopfield'

    """
    @summary: entraîne le réseau de neurone
    @param params: les paramètres de l'entraînement
    'dataset': le jeu d'apprentissage
    @type params: dictionary
    @param verbose: vrai si les valeurs des poids doivent s'afficher, faux sinon
    @type verbose: boolean
    @param visu: vrai si un affichage graphique du réseau doit être fait, faux sinon
    @type visu: boolean
    """
    def train(self, params, verbose, visu):
        if verbose:
            print("poids avant apprentissage\n",self.connection_weight)
        if visu:
            self.visu.show("avant apprentissage")
        dataset = params['dataset']
        dataset_size = dataset.shape[0]
        for i in range(dataset_size):
            self.set_input(dataset[i%dataset_size])
            self.learn()
        if verbose:
            print("poids après apprentissage\n",self.connection_weight)
        if visu:
            self.visu.show("après apprentissage")


if __name__=='__main__':
# Hebb
#    nb_neurons = 11
#    activation_function = activation_function_tanh
##    param_learn = {'learning_rate':0.1}
#    param_learn = {'learning_rate':0.1, 'vmin':-2,'vmax':2}
#    param_connection = {'percentage_connection':80}
#    param_weight = {'vmin':-1,'vmax':1}
##    ann = Hebb_ANN(nb_neurons, activation_function,param_learn,param_connection,param_weight)
#    ann = Hebb_Bounded_ANN(nb_neurons, activation_function,param_learn,param_connection,param_weight)
#    visu = Graph_ANN(ann)
#    ann.train({'epochs':80},True,True)
#    dataset = np.identity(nb_neurons)
#    ann.test(dataset)

# Oja
#    nb_neurons = 11
#    activation_function = activation_function_tanh
#    param_learn = {'learning_rate':0.1}
#    param_connection = {'percentage_connection':80}
#    param_weight = {'vmin':-1,'vmax':1}
#    ann = Oja_ANN(nb_neurons, activation_function,param_learn,param_connection,param_weight)
#    visu = Graph_ANN(ann)
#    ann.train({'epochs':80},True,True)
#    dataset = np.identity(nb_neurons)
#    ann.test(dataset)


# Oja (PCA)
#    nb_neurons = 5
#    param_learn = {'learning_rate':0.01}
#    param_weight = {'vmin':-1,'vmax':1}
#    ann = Oja_PCA_ANN(nb_neurons,param_learn,param_weight)
#    visu = Graph_ANN(ann)
#    mean = np.zeros(nb_neurons-1)
#    var = np.random.rand(nb_neurons-1,)
#    print("variance suivant chacun des axes: ",var)
#    covar = np.diag(var)
#    dataset_train = np.random.multivariate_normal(mean,covar,100)
#    ann.train({'dataset':dataset_train,'epochs':5000},True,True)

# Hopfield
    nb_neurons = 6
    nb_examples = 3
    dataset_train = np.array([[1,-1,1,-1,1,-1],[1,1,1,-1,-1,-1],[1,-1,-1,-1,1,1]])
    param_learn = {'learning_rate':1./nb_examples}
    ann = Hopfield_ANN(nb_neurons, param_learn)
    visu = Graph_ANN(ann)
    ann.train({'dataset':dataset_train},True,True)
    dataset = np.identity(nb_neurons)
    ann.test(dataset)

