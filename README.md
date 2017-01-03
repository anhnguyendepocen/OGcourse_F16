# MACS 40000: Economic Policy Analysis with Overlapping Generations Models (Autumn 2016) #

|  | [Dr. Richard Evans](https://sites.google.com/site/rickecon/) |
|--------------|--------------------------------------------------------------|
| Email | rwevans@uchicago.edu |
| Office | 250 Saieh Hall |
| Office Hours | W 2:30-4:30pm |
| GitHub | [rickecon](https://github.com/rickecon) |

* **Meeting day/time**: T,Th 10:30-11:50am, Saieh Hall, Room 102
* Office hours also available by appointment

## Prerequisites ##

Advanced undergraduate or first-year graduate microeconomic theory, linear algebra, multivariable calculus, recommended coding experience.


## Texts ##

* Evans, Richard W., _Overlapping Generations Models for Policy Analysis: Theory and Computation_, unpublished draft (2016).


## Course description ##

This course will study economic policy questions ideally addressed by the overlapping generations (OG) dynamic general equilibrium framework. OG models represent a rich class of macroeconomic general equilibrium model that is extremely useful for answering questions in which inequality, demographics, and individual heterogeneity are important. OG models are used extensively by the Joint Committee on Taxation, Congressional Budget Office, Department of the Treasury, and Federal Reserve System for policy analysis.

This course will train students how to set up and solve OG models. The standard nonlinear global solution method for these models--time path iteration--is a fixed point method that is similar to but significantly different from value function iteration and policy function iteration. This course will take students through progressively richer versions of the model, which will include endogenous labor supply, nontrivial demographics, bequests, stochastic income, multiple industries, non-balanced government budget constraint, and household tax structure.

We will be focusing on computational strategies, modularity of code, sensitivity and robustness to assumptions, and policy questions that can be answered with this framework. Students can use whatever programming language they want, but I highly recommend you use Python 3.x ([Anaconda distribution](https://www.continuum.io/downloads)). I will be most helpful with code debugging and suggestions in Python. We will also study results and uses from recent papers listed in the "References" section below. The dates on which we will be covering those references are listed in the "Daily Course Outline" section below.


## Course Objectives and Learning Outcomes ##

* You will know how to augment the standard overlapping generations model framework to answer policy questions.
* You will know the computational methods to solve for the steady-state and transition path equilibria of the model.
* You will learn computational tools such as:
	* Constrained minimization
	* Multi-equation root finder with occasionally binding constraints
	* Interpolation
	* Curve fitting
	* Parametric and nonparametric estimation
* You will learn coding and collaboration techniques such as:
	* Best practices for Python coding ([PEP 8](https://www.python.org/dev/peps/pep-0008/))
	* Writing modular code with functions and objects
	* Creating clear docstrings for functions
	* Collaboration tools for writing code using [Git](https://git-scm.com/) and [GitHub.com](https://github.com/).


## Grades ##

Grades will be based on the four categories listed below with the corresponding weights.

Assignment   | Points | Percent |
-------------|--------|------|
Homework     |   100  |  50% |
Midterm      |    50  |  25% |
Final Exam   |    50  |  25% |
**Total points** | **200** | **100%** |

* **Homework:** I will assign 7 problem sets throughout the term, and I will drop your one lowest problem set score.
	* You must write and submit your own computer code, although I encourage you to collaborate with your fellow students. I **DO NOT** want to see a bunch of copies of identical code. I **DO** want to see each of you learning how to code these problems so that you could do it on your own.
	* Problem set solutions, both written and code portions, will be turned in via a pull request from your private [GitHub.com](https://git-scm.com/) repository which is a fork of the class master repository on my account. (You will need to set up a GitHub account if you do not already have one.)
	* Problem sets will be due on the day listed in the Daily Course Outline section of this syllabus (see below) unless otherwise specified. Late homework will not be graded.
* **Midterm:** The midterm will be given on Thursday, October 27, during class and will cover the material up to that point in the course.
* **Final Exam:** The final exam will be comprehensive and will be given on Tuesday, Dec. 6, from 10:30a.m. to 12:30p.m. in our classroom (Saieh 102).


## Daily Course Schedule ##

| Date | Day | Topic | Readings | Homework |
|------|-----|-------|----------|----------|
| Sep. 27 |  T | Python, Git, OG Models        | Chs. 1, 2   |      |
|         |    |                               | Weil (2008) |      |
|         |    |                               | N&S (2007)  |      |
| Sep. 29 | Th | 3-period-lived model: theory  | Ch. 5       | PS 1 |
| Oct.  4 |  T | 3-period-lived model: theory  | Ch. 5       |      |
| Oct.  6 | Th | 3-period-lived model: computation | Ch. 5   |      |
| Oct. 11 |  T | *S*-period-lived model        | Ch. 6       | PS 2 |
| Oct. 13 | Th | Endogenous labor supply       | Ch. 7       |      |
|         |    |                               | P (2016)    |      |
|         |    |                               | E&P (2016)  |      |
| Oct. 18 |  T | Endogenous labor supply       | Ch. 7       | PS 3 |
| Oct. 20 | Th | Endogenous labor supply       | Ch. 7       |      |
| Oct. 25 |  T | Endogenous labor supply       | Ch. 7       |      |
| **Oct. 27** | **Th** | **Midterm 1 (Chs. 1-8)** |          | PS 4 |
| Nov.  1 |  T | Bequests: simple              | Ch. 10      |      |
|         |    |                               | D (2015)    |      |
|         |    |                               | DEMPRS (2016) |    |
| Nov.  3 | Th | Bequests: calibrated from SCF | Ch. 10      |      |
| Nov.  8 | T  | Population demographics       |  Ch. 9      | PS5  |
|         |    |                               | N (2015)    |      |
| Nov. 10 | Th | Population demographics       |  Ch. 9      |      |
| Nov. 15 | T  | Population demographics       |  Ch. 9      |      |
| Nov. 17 | Th | Tax functions from microsimulation |  Ch. 12 |     |
|         |    |                               | DEP (2016)  |      |
| Nov. 22 | T  | Tax functions from microsimulation | Ch. 12 | PS6  |
| Nov. 29 |  T | Tax functions from microsimulation | Ch. 12 |      |
| Dec.  1 | Th |                               |             | PS 7 |
|         |    | *Exam preparation (reading) days, Dec. 1-2* |  |   |
| **Dec. 6** | **T** | **Final Exam (comprehensive)** |      |      |
|         |     | **10:30a.m.-12:30p.m. in Saieh 102** |           |      |


## References ##

* De Nardi, Mariacristina, "[Quantitative Models of Wealth Inequality: A Survey](http://users.nber.org/~denardim/research/NBERwp21106.pdf)," National Bureau of Economic Research, NBER Working Paper 21106 (April 2015).
* DeBacker, Jason, Richard W. Evans, Evan Magnusson, Kerk L. Phillips, Shanthi Ramnath, and Isaac Swift, "[The Distributional Effects of Redistributional Tax Policy](https://sites.google.com/site/rickecon/WealthTax.pdf)," under review at *Quantitative Economics* (August 2016).
* DeBacker, Jason, Richard W. Evans, and Kerk L. Phillips, "[Integrating Microsimulation Tax Functions into a DGE Macroeconomic Model: A Canonical Example](https://sites.google.com/site/rickecon/DEP_10pct.pdf)," mimeo (August 2016).
* Evans, Richard W., *Overlapping Generations Models for Policy Analysis: Theory and Computation*, unpublished draft (2016).
* Evans, Richard W. and Kerk L. Phillips, "[Advantages of an Ellipse when Modeling Leisure Utility](https://sites.google.com/site/rickecon/Elliptical.pdf)," *Computational Economics*, (forthcoming, 2016).
* Nishiyama, Shinichi, "Fiscal Policy Effects in a Heterogeneous-agent OLG economy with an Aging Population," *Journal of Economic Dynamics & Control*, 61, pp. 114-132 (December 2015).
* Nishiyama, Shinichi and Kent Smetters, "Does Social Security Privatization Produce Efficiency Gains?," *Quarterly Journal of Economics*, 122:4, pp. 1677-1719 (November 2007).
* Peterman, William, "Reconciling Micro and Macro Estimates of the Frisch Labor Supply Elasticity," *Economic Inquiry*, 54:1, pp. 100-120 (January 2016).
* Weil, Philippe, "Overlapping Generations: The First Jubilee," *Journal of Economic Perspectives*, 22(4), 115-134 (Fall 2008).


## Disability services ##

If you need any special accommodations, please provide us with a copy of your Accommodation Determination Letter (provided to you by the Student Disability Services office) as soon as possible so that you may discuss with me how your accommodations may be implemented in this course.
