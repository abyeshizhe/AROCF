clc
clear all
close all

addpath(genpath('./utils'))
dsPath = './data/';

ds = {'Cora_ML_uni','Cora_OS_uni','20news_uni','text1_uni','TDT2_10_uni','SearchSnippets-lite','StackOverflow-lite'};
dataName = 'Cora_OS_uni'
load(strcat(dsPath,dataName));

% X=fea;
% Y=gnd;

X = (mapstd(X))';   % mpstd(X) should ensure the shape of X is [n,d] 

RESULT = [];
%% para
% anchor number 
m = 10;
% free para
lmd = 1;

     
tic
% Our method
[H,U,OBJ] = AROCF(X,Y,m,lmd);
t = toc;
[maxv,ind]=max(H,[],2);
[result] = ClusteringMeasure(Y, ind);

[result,t]
      






            
        


