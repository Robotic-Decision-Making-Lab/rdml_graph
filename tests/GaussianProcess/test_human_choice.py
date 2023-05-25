# test_user_gp_probits.py
# Written Ian Rankin - October 2022
#
# Test the probit functions for the user GP

import pytest
import rdml_graph as gr

import numpy as np




def test_human_pdf_single_value():
    r = np.array([0,0,0,500,0,0,0])

    true_pdf = np.array([0,0,0,1,0,0,0])
    pdf = gr.p_human_choice(r)

    tol = 0.0001
    for i in range(len(r)):
        assert true_pdf[i] + tol > pdf[i]
        assert true_pdf[i] - tol < pdf[i]

def test_human_pdf_same_value():
    r = np.array([5,5,5,5,5,5,5,5])

    true_pdf = np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125])
    pdf = gr.p_human_choice(r)

    tol = 0.0001
    for i in range(len(r)):
        assert true_pdf[i] + tol > pdf[i]
        assert true_pdf[i] - tol < pdf[i]


def test_human_sample_single_value():
    r = np.array([500,0,0])

    sample = gr.sample_human_choice(r)
    assert sample == 0

def test_human_sample_single_many():
    r = np.array([500,0,0])

    samples = gr.sample_human_choice(r, samples=600)

    for s in samples:
        assert s == 0


def test_human_sample_same():
    r = np.array([5,5,5,5,5,5,5,5])

    samples = gr.sample_human_choice(r, samples=20)

    assert len(samples) == 20
    for s in samples:
        assert s >= 0
        assert s < 8

