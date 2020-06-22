from enum import IntEnum
from typing import Tuple

import numpy as np

def get_ego_positions_from_scene(nusc, scene):
    '''Returns a 2xN array of the positions of the TOP LIDAR'''
    n_samples = scene['nbr_samples']

    pose_array = np.zeros(shape=(2, n_samples))
    i_sample = 0

    current_sample_token = scene['first_sample_token']
    current_sample = nusc.get('sample',current_sample_token)
    ego_pose_token = nusc.get('sample_data',current_sample['data']['LIDAR_TOP'])['ego_pose_token']
    ego_pose = nusc.get('ego_pose',ego_pose_token)
    pose_array[0,i_sample] = ego_pose['translation'][0]
    pose_array[1,i_sample] = ego_pose['translation'][1]

    while current_sample['next'] != '':
        i_sample += 1
        current_sample_token = current_sample['next']
        current_sample = nusc.get('sample',current_sample_token)
        ego_pose_token = nusc.get('sample_data',current_sample['data']['LIDAR_TOP'])['ego_pose_token']
        ego_pose = nusc.get('ego_pose',ego_pose_token)
        pose_array[0,i_sample] = ego_pose['translation'][0]
        pose_array[1,i_sample] = ego_pose['translation'][1]
    return pose_array

def compute_curvature_from_array(pose_array):
    '''Computes the curvature k from a 2xN array of ego poses.  Returns a 1xN array of curvature at each pt'''
    dx = np.gradient(pose_array[0,:])
    dy = np.gradient(pose_array[1,:])
    dydx = dy/dx
    dy2dx2 = np.gradient(dydx)/dx
    curvature = dy2dx2/((1 + dydx**2)**(3.0/2.0))
    return curvature

def query_scenes_for_word(nusc, word, query_phrase = False):
    ''' Query all scenes for a word.  Returns a list of tokens corresponding to the word queried 
        If query_phrase = True, then we will query for labeled phrases instead of words
    
    '''
    queried_scene_tokens = set([])
    
    for scene in nusc.scene:
        set_phrases = set(scene['description'].split(', '))
        
        set_words = set([])
        set_lower_phrases = set([])
        for phrase in set_phrases:
            set_lower_phrases.add(phrase.lower())
            
            words = phrase.split(' ')
            set_words |= set([w.lower() for w in words])
        
        if query_phrase:
            if word in set_lower_phrases:
                queried_scene_tokens.add(scene['token'])
        else:
            if word in set_words:
                queried_scene_tokens.add(scene['token'])
            
    return list(queried_scene_tokens)