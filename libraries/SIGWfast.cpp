// This file is part of SIGWfast.// SIGWfast is free software: you can use, copy, modify, merge, publish and// distribute, sublicense, and/or sell copies of it, and to permit persons to// whom it is furnished to do so it under the terms of the MIT License.// SIGWfast is distributed in the hope that it will be useful,// but WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT // LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR // PURPOSE AND NONINFRINGEMENT. See the MIT License for more details.// You should have received a copy of the MIT License along with SIGWfast. If// not, see <https://spdx.org/licenses/MIT.html>.#include <Python.h>#include "numpy/arrayobject.h"double *pyvector_to_Carrayptrs(PyArrayObject *arrayin) {	return (double *) arrayin->data;  /* pointer to arrayin data as double */};/* Method to compute the integral giving Omega_GW with the interation over s    split into two integrations, one for each side of the divergence in the    integation kernel */static PyObject* sigwint_w(PyObject* self, PyObject* args){	PyArrayObject *pyInt1, *pyInt2, *pyd, *pys1, *pys2;	double *cInt1, *cInt2, *cd, *cs1, *cs2;   // The C vectors to be created 	                      // to point to the python vectors, cInt1, cInt2, cd,	                      // cs1, cs2 point to the row of pyInt1, pyInt2, pyd,	                      // pys1, pys2, respectively	                      	// Declare integers for looping and to hold the length of imported arrays	int i, j, nd, ns1, ns2;		/* Parse tuples separately since args will differ between C fcns */	if (!PyArg_ParseTuple(args, "O!O!O!O!O!iii", 		&PyArray_Type, &pyInt1, &PyArray_Type, &pyInt2,		&PyArray_Type, &pyd, &PyArray_Type, &pys1, &PyArray_Type, &pys2, 		&nd, &ns1, &ns2)) return NULL;	if (NULL == pyInt1) return NULL;	if (NULL == pyInt2) return NULL;	if (NULL == pyd) return NULL;	if (NULL == pys1) return NULL;	if (NULL == pys2) return NULL;			/* Change contiguous arrays into C *arrays   */	cInt1=pyvector_to_Carrayptrs(pyInt1);	cInt2=pyvector_to_Carrayptrs(pyInt2);	cd=pyvector_to_Carrayptrs(pyd);	cs1=pyvector_to_Carrayptrs(pys1);	cs2=pyvector_to_Carrayptrs(pys2);		/* Declare variables to hold intermediate and final results of the 	   discrete integration  */	double cId[nd], cI;		/* Do the calculation: Two nested discrete integrations using the 	   trapezoidal rule. The innermost integration is split into two	   independent integrations */	cI=0;	for (i=0; i<nd; i++)  {    	cId[i]=0;    	for (j=i*ns1+1; j<(i+1)*ns1; j++)  {    			cId[i]= cId[i]+(cInt1[j]+cInt1[j-1])*(cs1[j]-cs1[j-1])/2;    	};    	for (j=i*ns2+1; j<(i+1)*ns2; j++)  {    			cId[i]= cId[i]+(cInt2[j]+cInt2[j-1])*(cs2[j]-cs2[j-1])/2;    	};    	if( i == 0 ) {} else {cI= cI+(cId[i]+cId[i-1])*(cd[i]-cd[i-1])/2;};   		};    /* Export integration result*/    return Py_BuildValue("d",cI);};/* Method to compute the integral giving Omega_GW for the case that the kernel   does not exhibit any divergence */static PyObject* sigwint_1(PyObject* self, PyObject* args){	PyArrayObject *pyInt, *pyd, *pys;	double *cInt, *cd, *cs;   // The C vectors to be created to point to the 	                      // python vectors, cInt, cd, cs point to the row	                      // of pyInt, pyd, pys, respectively	                      	// Declare integers for looping and to hold the length of imported arrays                   	int i, j, nd, ns;		/* Parse tuples separately since args will differ between C fcns */	if (!PyArg_ParseTuple(args, "O!O!O!ii", 		&PyArray_Type, &pyInt, &PyArray_Type, &pyd,		&PyArray_Type, &pys, &nd, &ns))  return NULL;	if (NULL == pyInt) return NULL;	if (NULL == pyd) return NULL;	if (NULL == pys) return NULL;			/* Change contiguous arrays into C *arrays   */	cInt=pyvector_to_Carrayptrs(pyInt);	cd=pyvector_to_Carrayptrs(pyd);	cs=pyvector_to_Carrayptrs(pys);		double cId[nd], cI;		/* Do the calculation: Two nested discrete integrations using the 	   trapezoidal rule */	cI=0;	for (i=0; i<nd; i++)  {    	cId[i]=0;    	for (j=i*ns+1; j<(i+1)*ns; j++)  {    			cId[i]= cId[i]+(cInt[j]+cInt[j-1])*(cs[j]-cs[j-1])/2;    	};    	if( i == 0 ) {} else {cI= cI+(cId[i]+cId[i-1])*(cd[i]-cd[i-1])/2;};   		};    /* Export integration result*/    return Py_BuildValue("d",cI);};/* List the methods defined above, declare their names in the compiled    Python module and give a brief description */static PyMethodDef sigwfastMethods[] = { {"sigwint_w",sigwint_w,METH_VARARGS,"Compute SIGW for c_s^2=w"}, {"sigwint_1",sigwint_1,METH_VARARGS,"Compute SIGW for c_s^2=1"}, {NULL,NULL,0,NULL}};/* Declatrations needed to compile the Python module. Do not change */static PyModuleDef sigwfast = { PyModuleDef_HEAD_INIT, "sigwfast","SIGW computation", -1, sigwfastMethods};PyMODINIT_FUNC PyInit_sigwfast(void){	PyObject *m;	import_array();		m = PyModule_Create(&sigwfast);	if(m == NULL){		return NULL;	}	return m;}