This folder contains the code for testing feature, we use this code to confine that after the change of the code, the other version code such as v5, v7 will not be confluenced.

----previous prepare:

we use the pytest framework. And you need to run the below command line:

pip install -U pytest


----test method:

after installing the pytest. You can run the below command to test all the feature:

pytest -v -s pass_test.py


----test specific method:

if you want to test the specific method:

--only test the test_one:

pytest -v -s pass_test.py::test_one

--only test the test_two: 

pytest -v -s pass_test.py::test_two
