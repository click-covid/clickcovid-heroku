# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile
import kaggle
import os
from scipy.integrate import odeint

def f(tr,c,td,re,Pb,Ef,tc,tf,ifr):
        def Corona6(x,t):
                S1 = x[0]
                E1 = x[1]
                I1 = x[2]
                R1 = x[3]
                M1 = x[4]
            
                S2 = x[5]
                E2 = x[6]
                I2 = x[7]
                R2 = x[8]
                M2 = x[9]
            
                S3 = x[10]
                E3 = x[11]
                I3 = x[12]
                R3 = x[13]
                M3 = x[14]
            
                S4 = x[15]
                E4 = x[16]
                I4 = x[17]
                R4 = x[18]
                M4 = x[19]
            
                S5 = x[20]
                E5 = x[21]
                I5 = x[22]
                R5 = x[23]
                M5 = x[24]
            
                S6 = x[25]
                E6 = x[26]
                I6 = x[27]
                R6 = x[28]
                M6 = x[29]
            
                S7 = x[30]
                E7 = x[31]
                I7 = x[32]
                R7 = x[33]
                M7 = x[34]
            
                S8 = x[35]
                E8 = x[36]
                I8 = x[37]
                R8 = x[38]
                M8 = x[39]
            
                N = 23.6e6
                N1 = 0.1367*N
                N2 = 0.1465*N
                N3 = 0.1604*N
                N4 = 0.1621*N
                N5 = 0.1386*N
                N6 = 0.1148*N
                N7 = 0.0790*N
                N8 = 0.0610*N
                N = N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8
                        
                #MATRIZ DA POLONIA
                Cij = np.array([
                [3.21, 0.88, 0.96, 1.39, 0.58, 0.80, 0.43, 0.33],
                [0.56, 8.46, 1.05, 0.96, 0.89, 0.46, 0.26, 0.28],
                [0.50, 0.86, 4.92, 1.33, 1.06, 1.08, 0.26, 0.22],
                [0.91, 0.98, 1.65, 2.89, 1.68, 1.20, 0.54, 0.40],
                [0.35, 0.84, 1.24, 1.57, 2.22, 1.19, 0.42, 0.43],
                [0.48, 0.44, 1.24, 1.12, 1.18, 1.80, 0.53, 0.50],
                [0.44, 0.42, 0.52, 0.86, 0.71, 0.91, 0.91, 0.47],
                [0.33, 0.44, 0.41, 0.61, 0.70, 0.81, 0.45, 0.90]
                ])
            
            
                a = 1/tr
                #c = 1/5.2
                mu = 1/td
                d = 1/re
                #Pb = 0.11
                Pk = Pb*0.44
                Ej = np.array([E1,E2,E3,E4,E5,E6,E7,E8])
                Ij = np.array([I1,I2,I3,I4,I5,I6,I7,I8])
                B1j = sum(Cij[0,:]*Ij)
                k1j = sum(Cij[0,:]*Ej)
            
                B1 = Ef*(B1j*Pb/N1)/(1+0.01*np.exp(t-tc)) + (1-Ef)*(B1j*Pb/N1) + Ef*(B1j*Pb/N1)/(1+100*np.exp(-t+tf))
                c1 = 1/c
                a1 = (1-0.00143*ifr)*a
                mu1 = 0.00143*ifr**mu
                k1 = k1j*Pk/N1 
                dS1dt = -B1*S1 - k1*S1 + d*R1
                dE1dt = B1*S1 - c1*E1 + k1*S1
                dI1dt = c1*E1 - a1*I1 - mu1*I1
                dR1dt = a1*I1 - d*R1
                dM1dt = mu1*I1
            
            
                B2j = sum(Cij[1,:]*Ij)
                k2j = sum(Cij[1,:]*Ej)
            
                B2 = Ef*(B2j*Pb/N2)/(1+0.01*np.exp(t-tc)) + (1-Ef)*(B2j*Pb/N2) + Ef*(B2j*Pb/N2)/(1+100*np.exp(-t+tf))
                c2 = 1/c
                a2 = (1-0.00143*ifr)*a
                mu2 = 0.00143*ifr**mu
                k2 = k2j*Pk/N2
                dS2dt = -B2*S2 - k2*S2 + d*R2
                dE2dt = B2*S2 - c2*E2 + k2*S2
                dI2dt = c2*E2 - a2*I2 - mu2*I2
                dR2dt = a2*I2 - d*R2
                dM2dt = mu2*I2
            
            
                B3j = sum(Cij[2,:]*Ij)
                k3j = sum(Cij[2,:]*Ej)
            
                B3 = Ef*(B3j*Pb/N3)/(1+0.01*np.exp(t-tc)) + (1-Ef)*(B3j*Pb/N3) + Ef*(B3j*Pb/N3)/(1+100*np.exp(-t+tf))
                c3 = 1/c
                a3 = (1-0.0713*ifr)*a
                mu3 = 0.0713*ifr*mu
                k3 = k3j*Pk/N3
                dS3dt = -B3*S3 - k3*S3 + d*R3
                dE3dt = B3*S3 - c3*E3 + k3*S3
                dI3dt = c3*E3 - a3*I3 - mu3*I3
                dR3dt = a3*I3 - d*R3
                dM3dt = mu3*I3
                
                B4j = sum(Cij[3,:]*Ij)
                k4j = sum(Cij[3,:]*Ej)
            
                B4 = Ef*(B4j*Pb/N4)/(1+0.01*np.exp(t-tc)) + (1-Ef)*(B4j*Pb/N4) + Ef*(B4j*Pb/N4)/(1+100*np.exp(-t+tf))
                c4 = 1/c
                a4 = (1-0.0786*ifr)*a
                mu4 = 0.0786*ifr*mu
                k4 = k4j*Pk/N4 
                dS4dt = -B4*S4 - k4*S4 + d*R4
                dE4dt = B4*S4 - c4*E4 + k4*S4
                dI4dt = c4*E4 - a4*I4 - mu4*I4
                dR4dt = a4*I4 - d*R4
                dM4dt = mu4*I4
            
            
                B5j = sum(Cij[4,:]*Ij)
                k5j = sum(Cij[4,:]*Ej)
            
                B5 = Ef*(B5j*Pb/N5)/(1+0.01*np.exp(t-tc)) + (1-Ef)*(B5j*Pb/N5) + Ef*(B5j*Pb/N5)/(1+100*np.exp(-t+tf))
                c5 = 1/c
                a5 = (1-0.1714*ifr)*a
                mu5 = 0.1714*ifr*mu
                k5 = k5j*Pk/N5 
                dS5dt = -B5*S5 - k5*S5 + d*R5
                dE5dt = B5*S5 - c5*E5 + k5*S5
                dI5dt = c5*E5 - a5*I5 - mu5*I5
                dR5dt = a5*I5 - d*R5
                dM5dt = mu5*I5
            
            
                B6j = sum(Cij[5,:]*Ij)
                k6j = sum(Cij[5,:]*Ej)
            
                B6 = Ef*(B6j*Pb/N6)/(1+0.01*np.exp(t-tc)) + (1-Ef)*(B6j*Pb/N6) + Ef*(B6j*Pb/N6)/(1+100*np.exp(-t+tf))
                c6 = 1/c
                a6 = (1-1.714*ifr)*a
                mu6 = 1.714*ifr*mu
                k6 = k6j*Pk/N6 
                dS6dt = -B6*S6 - k6*S6 + d*R6
                dE6dt = B6*S6 - c6*E6 + k6*S6
                dI6dt = c6*E6 - a6*I6 - mu6*I6
                dR6dt = a6*I6 - d*R6
                dM6dt = mu6*I6
            
            
                B7j = sum(Cij[6,:]*Ij)
                k7j = sum(Cij[6,:]*Ej)
            
                B7 = Ef*(B7j*Pb/N7)/(1+0.01*np.exp(t-tc)) + (1-Ef)*(B7j*Pb/N7) + Ef*(B7j*Pb/N7)/(1+100*np.exp(-t+tf))
                c7 = 1/c
                a7 = (1-3.714*ifr)*a
                mu7 = 3.714*ifr*mu
                k7 = k7j*Pk/N7 
                dS7dt = -B7*S7 - k7*S7 + d*R7
                dE7dt = B7*S7 - c7*E7 + k7*S7
                dI7dt = c7*E7 - a7*I7 - mu7*I7
                dR7dt = a7*I7 - d*R7
                dM7dt = mu7*I7
            
            
                B8j = sum(Cij[7,:]*Ij)
                k8j = sum(Cij[7,:]*Ej)
            
                B8 = Ef*(B8j*Pb/N8)/(1+0.01*np.exp(t-tc)) + (1-Ef)*(B8j*Pb/N8) + Ef*(B8j*Pb/N8)/(1+100*np.exp(-t+tf))
                c8 = 1/c
                a8 = (1-14.286*ifr)*a
                mu8 = 14.286*ifr*mu
                k8 = k8j*Pk/N8 
                dS8dt = -B8*S8 - k8*S8 + d*R8
                dE8dt = B8*S8 - c8*E8 + k8*S8
                dI8dt = c8*E8 - a8*I8 - mu8*I8
                dR8dt = a8*I8 - d*R8
                dM8dt = mu8*I8
                
                return [dS1dt, dE1dt, dI1dt, dR1dt, dM1dt,
                        dS2dt, dE2dt, dI2dt, dR2dt, dM2dt,
                        dS3dt, dE3dt, dI3dt, dR3dt, dM3dt,
                        dS4dt, dE4dt, dI4dt, dR4dt, dM4dt,
                        dS5dt, dE5dt, dI5dt, dR5dt, dM5dt,
                        dS6dt, dE6dt, dI6dt, dR6dt, dM6dt,
                        dS7dt, dE7dt, dI7dt, dR7dt, dM7dt,
                        dS8dt, dE8dt, dI8dt, dR8dt, dM8dt]

        N = 23.6e6
        N1 = 0.137*N
        N2 = 0.146*N
        N3 = 0.160*N
        N4 = 0.162*N
        N5 = 0.139*N
        N6 = 0.115*N
        N7 = 0.080*N
        N8 = 0.061*N
        I10 = 0
        I20 = 0
        I30 = 0
        I40 = 1
        I50 = 0
        I60 = 0
        I70 = 2
        I80 = 1


        x0 = [N1,0,I10,0,0,
              N2,0,I20,0,0,
              N3,0,I30,0,0,
              N4,0,I40,0,0,
              N5,0,I50,0,0,
              N6,0,I60,0,0,
              N7,0,I70,0,0,
              N8,0,I80,0,0]
        t = np.linspace(1, 800, 1000)
        x = odeint(Corona6, x0, t)

        S1 = x[:,0]
        E1 = x[:,1]
        I1 = x[:,2]
        R1 = x[:,3]
        M1 = x[:,4]
        CT1 = M1 + R1 + I1
        S2 = x[:,5]
        E2 = x[:,6]
        I2 = x[:,7]
        R2 = x[:,8]
        M2 = x[:,9]
        CT2 = M2 + R2 + I2
        S3 = x[:,10]
        E3 = x[:,11]
        I3 = x[:,12]
        R3 = x[:,13]
        M3 = x[:,14]
        CT3 = M3 + R3 + I3
        S4 = x[:,15]
        E4 = x[:,16]
        I4 = x[:,17]
        R4 = x[:,18]
        M4 = x[:,19]
        CT4 = M4 + R4 + I4
        S5 = x[:,20]
        E5 = x[:,21]
        I5 = x[:,22]
        R5 = x[:,23]
        M5 = x[:,24]
        CT5 = M5 + R5 + I5
        S6 = x[:,25]
        E6 = x[:,26]
        I6 = x[:,27]
        R6 = x[:,28]
        M6 = x[:,29]
        CT6 = M6 + R6 + I6
        S7 = x[:,30]
        E7 = x[:,31]
        I7 = x[:,32]
        R7 = x[:,33]
        M7 = x[:,34]
        CT7 = M7 + R7 + I7
        S8 = x[:,35]
        E8 = x[:,36]
        I8 = x[:,37]
        R8 = x[:,38]
        M8 = x[:,39]
        CT8 = M8 + R8 + I8

        I = I1 + I2 + I3 + I4 + I5 + I6 + I7 + I8
        M = M1 + M2 + M3 + M4 + M5 + M6 + M7 + M8
        CT = CT1 + CT2 + CT3 + CT4 + CT5 + CT6 + CT7 + CT8

        #A = [[I],[M],[CT],[t]]
        
        return [I,M,CT,t]
	
	
############################


graph1 = go.Scatter(
	x=[0,0,0],
	y=[0,0,0],
	name = "Infectados"
)

graph2 = go.Scatter(
	x=[0,0,0],
	y=[0,0,0],
	name = "Mortos"
)

graph3 = go.Scatter(
	x=[0,0,0],
	y=[0,0,0],
	name = "Casos graves cumulativos"
)



dat = [graph1,graph2,graph3]

beer_fig = go.Figure(data=dat)

beer_fig.update_layout(
    title="Modelo SEIRDS com divisao de idades",
    xaxis_title="Tempo (dias)",
    yaxis_title="Numero de casos",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)

kaggle.api.dataset_download_files(r"unanimad/corona-virus-brazil", "corona-virus-brazil")

zipp = ZipFile(r"corona-virus-brazil/corona-virus-brazil.zip")

st_dat = pd.read_csv(zipp.open("brazil_covid19.csv"))

ct_dat = pd.read_csv(zipp.open("brazil_covid19_cities.csv"))

os.system(r"rm -rf corona-virus-brazil")
del(zipp)

states = np.sort(st_dat.state.unique())

app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True
server = app.server
app.title = "Click Covid"

layout_home = html.Div(style={"width":"100vw", "margin-left":-8, "margin-top":-10}, children=[
	html.Div(style={"height":70, "width":"100%", "background-color":"#F29C04", "display":"flex"}, children=[
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":238, "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.Img(src="/assets/click_covid.jpg", style={"height":70, "margin":{"l":0, "r":0, "t":0, "b":0}})], href="/")
			#~ dcc.Link([html.H2("Click Covid")], href="/", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"10%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Cidades")], href="/page-cities", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"}, children=[
			html.A([html.H3("Simulação")], href="/page-simulation", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"}, children=[
			html.A([html.H3("Análise Socioeconômica")], href="/page-socio", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Equipe")], href="/page-equipe", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Agradecimentos")], href="/page-thanks", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			html.A([html.H3("GitHub")], href="https://github.com/click-covid/clickcovid-heroku", target="_blank", className="bar"),
		]),
	]),
	#~ html.Div(style={"background-image":'url("/assets/corona_cinza_fundo_Prancheta_1.jpg")', "background-size":"cover", "width":"100vw", "height":"85vh", "margin-left":-8})
])

graph_states = {"data": [{"x":st_dat[st_dat.state==states[0]].date, "y":st_dat[st_dat.state==states[0]].cases, "name":"Casos", "showlegend":True}, {"x":st_dat[st_dat.state==states[0]].date, "y":st_dat[st_dat.state==states[0]].deaths, "name":"Mortes", "showlegend":True}], "layout":{"width":"100%", "margin":{"l":30, "r":0, "t":20, "b":30}}}

st = ct_dat[ct_dat.state==states[0]]
stcts = st.name.unique()

graph_cities = {"data": [{"x":st_dat[st_dat.state=="São Paulo"].date, "y":st_dat[st_dat.state=="São Paulo"].cases, "name":"Casos", "showlegend":True}, {"x":st_dat[st_dat.state=="São Paulo"].date, "y":st_dat[st_dat.state=="São Paulo"].deaths, "name":"Mortes", "showlegend":True}], "layout":{"width":"100%", "margin":{"l":30, "r":0, "t":20, "b":30}}}

layout_cities = html.Div(style={"width":"100vw", "margin-left":-8, "margin-top":-10}, children=[
	html.Div(style={"height":70, "width":"100%", "background-color":"#F29C04", "display":"flex"}, children=[
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":238, "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.Img(src="/assets/click_covid.jpg", style={"height":70, "margin":{"l":0, "r":0, "t":0, "b":0}})], href="/")
			#~ dcc.Link([html.H2("Click Covid")], href="/", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"10%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Cidades")], href="/page-cities", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"}, children=[
			html.A([html.H3("Simulação")], href="/page-simulation", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"}, children=[
			html.A([html.H3("Análise Socioeconômica")], href="/page-socio", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Equipe")], href="/page-equipe", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Agradecimentos")], href="/page-thanks", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			html.A([html.H3("GitHub")], href="https://github.com/click-covid/clickcovid-heroku", target="_blank", className="bar"),
		]),
	]),
	html.Div(style={"height":20,}),	
	html.Div(style={"display":"flex"}, children=[
		html.Div(style={"width":"1%"}),
		html.Div(style={"width":"69%"}, children=[
			dcc.Graph(id='graph-cities', figure=graph_cities),
			html.P(["Fonte: ", html.A(["kaggle/unanimad/corona-virus-brazil"], href="https://www.kaggle.com/unanimad/corona-virus-brazil", target="_blank")]),
		]),
		html.Div(style={"width":"2%"}),
		html.Div(style={"width":"38%", "text-align":"center"}, children=[
			html.Div(style={"width":"100%", "display":"flex"}, children=[
				html.P(id='drop-out-states2', children=['Estado:'], style={"width":"20%"}),    
				dcc.Dropdown(
					id="drop-states2",
					options=[{'label': i, 'value': i} for i in states],
					value="São Paulo",
					style={"width":"80%"}
				),
			]),
			html.Div(style={"width":"100%", "display":"flex"}, children=[
				html.P(id='drop-out-cities', children=['Cidade:'], style={"width":"20%"}),    
				dcc.Dropdown(
					id="drop-cities",
					options=[{"label":"Todo o Estado", "value":0}] + [{'label': i, 'value': i} for i in stcts],
					value=0,
					style={"width":"80%"}
				),
			]),
		]),
	])
])

layout_socio = html.Div(style={"width":"100vw", "margin-left":-8, "margin-top":-10}, children=[
	html.Div(style={"height":70, "width":"100%", "background-color":"#F29C04", "display":"flex"}, children=[
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":238, "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.Img(src="/assets/click_covid.jpg", style={"height":70, "margin":{"l":0, "r":0, "t":0, "b":0}})], href="/")
			#~ dcc.Link([html.H2("Click Covid")], href="/", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"10%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Cidades")], href="/page-cities", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"}, children=[
			html.A([html.H3("Simulação")], href="/page-simulation", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"}, children=[
			html.A([html.H3("Análise Socioeconômica")], href="/page-socio", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Equipe")], href="/page-equipe", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Agradecimentos")], href="/page-thanks", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			html.A([html.H3("GitHub")], href="https://github.com/click-covid/clickcovid-heroku", target="_blank", className="bar"),
		]),
	]),
	
	html.Div(style={"height":20,}),	
	html.Div(style={"display":"flex"}, children=[
		html.Div(style={"width":"2%"}),
		html.Div(style={"width":"60%"}, children=[
			html.Img(src="/assets/Leitos_de_UTI_por_estado2.svg", style={"width":"100%"}),
		]),
		html.Div(style={"width":"2%"}),
		html.Div(style={"width":"20%"}, children=[
			html.P(["Fonte: ", html.A(["AMIB"], href="https://www.amib.com.br/censo/Analise_de_Dados_UTI_Final_Site_2.pdf", target="_blank"),]),
		]),
	]),
	
	html.Div(style={"height":20,}),	
	html.Div(style={"display":"flex"}, children=[
		html.Div(style={"width":"2%"}),
		html.Div(style={"width":"60%"}, children=[
			html.Img(src="/assets/Leitos_de_UTI_para_cada_10_mil_habitantes2.svg", style={"width":"100%"}),
		]),
		html.Div(style={"width":"2%"}),
		html.Div(style={"width":"20%"}, children=[
			html.P(["Fontes: ", html.A(["AMIB"], href="https://www.amib.com.br/censo/Analise_de_Dados_UTI_Final_Site_2.pdf", target="_blank"), "; ", html.A(["IBGE"], href="https://cidades.ibge.gov.br/brasil/pesquisa/23/25888", target="_blank"),]),
		]),
	]),
	
	html.Div(style={"height":20,}),	
	html.Div(style={"display":"flex"}, children=[
		html.Div(style={"width":"2%"}),
		html.Div(style={"width":"60%"}, children=[
			html.Img(src="/assets/Proporcao_da_populacao_abastecimento_de_agua_por_rede_geral2.svg", style={"width":"100%"}),
		]),
		html.Div(style={"width":"2%"}),
		html.Div(style={"width":"20%"}, children=[
			html.P(["Fonte: ", html.A(["IBGE"], href="https://www.ibge.gov.br/estatisticas/multidominio/condicoes-de-vida-desigualdade-e-pobreza/9221-sintese-de-indicadores-sociais.html?=&t=resultados&utm_source=covid19&utm_medium=hotsite&utm_campaign=covid_19", target="_blank"),]),
		]),
	]),
	
	html.Div(style={"height":20,}),	
	html.Div(style={"display":"flex"}, children=[
		html.Div(style={"width":"2%"}),
		html.Div(style={"width":"60%"}, children=[
			html.Img(src="/assets/Porcentagem_da_populacao_sem_acesso_a_saneamento_basico2.svg", style={"width":"100%"}),
		]),
		html.Div(style={"width":"2%"}),
		html.Div(style={"width":"20%"}, children=[
			html.P(["Fonte: ", html.A(["IBGE"], href="https://www.ibge.gov.br/estatisticas/sociais/trabalho/9171-pesquisa-nacional-por-amostra-de-domicilios-continua-mensal.html?=&t=o-que-e", target="_blank"),]),
		]),
	]),
	
	html.Div(style={"height":20,}),	
	html.Div(style={"display":"flex"}, children=[
		html.Div(style={"width":"2%"}),
		html.Div(style={"width":"60%"}, children=[
			html.Img(src="/assets/Adensamento_Excessivo_2.svg", style={"width":"100%"}),
		]),
		html.Div(style={"width":"2%"}),
		html.Div(style={"width":"20%"}, children=[
			html.P(["Fonte: ", html.A(["IBGE"], href="https://www.ibge.gov.br/estatisticas/sociais/populacao/9127-pesquisa-nacional-por-amostra-de-domicilios.html?=&t=o-que-e", target="_blank"),]),
		]),
	]),
	
	html.Div(style={"height":20,}),	
	html.Div(style={"display":"flex"}, children=[
		html.Div(style={"width":"2%"}),
		html.Div(style={"width":"60%"}, children=[
			html.Img(src="/assets/Porcentagem_da_populacao_vivendo_em_situacao_de_adensamento_excessivo2.svg", style={"width":"100%"}),
		]),
		html.Div(style={"width":"2%"}),
		html.Div(style={"width":"20%"}, children=[
			html.P(["Fonte: ", html.A(["IBG"], href="https://www.ibge.gov.br/estatisticas/sociais/trabalho/9171-pesquisa-nacional-por-amostra-de-domicilios-continua-mensal.html?=&t=o-que-e", target="_blank"),]),
		]),
	]),
])

layout_equipe = html.Div(style={"width":"100vw", "margin-left":-8, "margin-top":-10}, children=[
	html.Div(style={"height":70, "width":"100%", "background-color":"#F29C04", "display":"flex"}, children=[
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":238, "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.Img(src="/assets/click_covid.jpg", style={"height":70, "margin":{"l":0, "r":0, "t":0, "b":0}})], href="/")
			#~ dcc.Link([html.H2("Click Covid")], href="/", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"10%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Cidades")], href="/page-cities", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"}, children=[
			html.A([html.H3("Simulação")], href="/page-simulation", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"}, children=[
			html.A([html.H3("Análise Socioeconômica")], href="/page-socio", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Equipe")], href="/page-equipe", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Agradecimentos")], href="/page-thanks", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			html.A([html.H3("GitHub")], href="https://github.com/click-covid/clickcovid-heroku", target="_blank", className="bar"),
		]),
	]),
	html.Div(style={"height":20}),
	html.Div(style={"height":400, "width":"100vw", "margin-left":-8, "display":"flex"}, children=[
		html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":400, "width":"25vw", "text-align":"center"},
				children=[
					html.Img(src="/assets/Felipe_Fontinele.jpg", style={"width":200, "height":200}),
					html.H4(["Felipe Fontinele", html.Br(), "Universidade de Brasília"]),
					html.A([html.Img(src="/assets/ig.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="https://www.instagram.com/feradofogo/", target="_blank"),
					html.A([html.Img(src="/assets/lattes2.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="http://lattes.cnpq.br/1862967944333863", target="_blank")
				],
			),
		html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":400, "width":"25vw", "text-align":"center"},
				children=[
					html.Img(src="/assets/Igor_Reis.jpg", style={"width":200, "height":200}),
					html.H4(["Igor Reis", html.Br(), "Universidade de Brasília"]),
					html.A([html.Img(src="/assets/ig.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="https://www.instagram.com/igrorreis/", target="_blank"),
					html.A([html.Img(src="/assets/lattes2.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="http://lattes.cnpq.br/0168960446714554", target="_blank")
				],
			),
		html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":400, "width":"25vw", "text-align":"center"},
				children=[
					html.Img(src="/assets/Joao_Valeriano.jpg", style={"width":200, "height":200}),
					html.H4(["João Valeriano", html.Br(), "Universidade de Brasília"]),
					html.A([html.Img(src="/assets/ig.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="https://www.instagram.com/joaopedro_vm/", target="_blank"),
					html.A([html.Img(src="/assets/lattes2.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="http://lattes.cnpq.br/2757621633925765", target="_blank")
				],
			),
		html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":400, "width":"25vw", "text-align":"center"},
				children=[
					html.Img(src="/assets/Joao_Vitor.jpg", style={"width":200, "height":200}),
					html.H4(["João Vítor", html.Br(), "Universidade de Brasília"]),
					html.A([html.Img(src="/assets/ig.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="https://www.instagram.com/viitor_jvc/", target="_blank"),
					html.A([html.Img(src="/assets/lattes2.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="http://lattes.cnpq.br/0516093970342061", target="_blank")
				],
			),
	]
	),
	html.Div(style={"height":400, "width":"100vw", "margin-left":-8, "display":"flex"}, children=[
		html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":400, "width":"25vw", "text-align":"center"},
				children=[
					html.Img(src="/assets/Jose_Luiz.jpg", style={"width":200, "height":200}),
					html.H4(["José Luiz", html.Br(), "Universidade de Michigan"]),
					html.A([html.Img(src="/assets/ig.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="https://www.instagram.com/joseluiz_vargas/", target="_blank"),
					html.A([html.Img(src="/assets/lattes2.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="http://lattes.cnpq.br/4939980150228138", target="_blank")
				],
			),
		html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":400, "width":"25vw", "text-align":"center"},
				children=[
					html.Img(src="/assets/Ludmila_Lima.jpg", style={"width":200, "height":200}),
					html.H4(["Ludmila Lima", html.Br(), "Universidade de Brasília"]),
					html.A([html.Img(src="/assets/ig.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="https://www.instagram.com/Ludlima_m/", target="_blank"),
					html.A([html.Img(src="/assets/lattes2.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="http://lattes.cnpq.br/7173638953043122", target="_blank")
				],
			),
		html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":400, "width":"25vw", "text-align":"center"},
				children=[
					html.Img(src="/assets/Pedro_Cintra.jpg", style={"width":200, "height":200}),
					html.H4(["Pedro Cintra", html.Br(), "Universidade de Brasília"]),
					html.A([html.Img(src="/assets/ig.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="https://www.instagram.com/pedrocintra52/", target="_blank"),
					html.A([html.Img(src="/assets/lattes2.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="http://lattes.cnpq.br/1191661313631770", target="_blank")
				],
			),
		html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":400, "width":"25vw", "text-align":"center"},
				children=[
					html.Img(src="/assets/Tabata_Luiza.jpg", style={"width":200, "height":200}),
					html.H4(["Tábata Luiza", html.Br(), "Universidade de Brasília"]),
					html.A([html.Img(src="/assets/ig.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="https://www.instagram.com/batatalsa/", target="_blank"),
					html.A([html.Img(src="/assets/lattes2.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="http://lattes.cnpq.br/5177569135658501", target="_blank")
				],
			),
	]
	),
])

layout_thanks = html.Div(style={"width":"100vw", "margin-left":-8, "margin-top":-10}, children=[
	html.Div(style={"height":70, "width":"100%", "background-color":"#F29C04", "display":"flex"}, children=[
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":238, "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.Img(src="/assets/click_covid.jpg", style={"height":70, "margin":{"l":0, "r":0, "t":0, "b":0}})], href="/")
			#~ dcc.Link([html.H2("Click Covid")], href="/", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"10%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Cidades")], href="/page-cities", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"}, children=[
			html.A([html.H3("Simulação")], href="/page-simulation", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"}, children=[
			html.A([html.H3("Análise Socioeconômica")], href="/page-socio", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Equipe")], href="/page-equipe", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Agradecimentos")], href="/page-thanks", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			html.A([html.H3("GitHub")], href="https://github.com/click-covid/clickcovid-heroku", target="_blank", className="bar"),
		]),
	]),
	html.Div(style={"height":20}),
	html.Div(style={"height":400, "width":"100vw", "margin-left":-8, "display":"flex"}, children=[
		html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":400, "width":"25vw", "text-align":"center"},
				children=[
					html.Img(src="/assets/Lorena_Lima.jpg", style={"width":200, "height":200}),
					html.H4(["Lorena Lima", html.Br(), "Universidade de Brasília"]),
					html.A([html.Img(src="/assets/ig.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="https://www.instagram.com/_madame_satan/", target="_blank"),
					html.A([html.Img(src="/assets/lattes2.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="http://lattes.cnpq.br/3527099707324024", target="_blank")
				],
			),
		html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":400, "width":"25vw", "text-align":"center"},
				children=[
					html.Img(src="/assets/Victor_Lima.jpg", style={"width":200, "height":200}),
					html.H4(["Victor Lima", html.Br(), "Universidade de Brasília"]),
					html.A([html.Img(src="/assets/ig.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="https://www.instagram.com/victorportog_/", target="_blank"),
					html.A([html.Img(src="/assets/lattes2.png", style={"width":20, "height":20, "padding-left":5, "padding-right":5})], href="http://lattes.cnpq.br/5193576100573774", target="_blank")
				],
			),
	]
	),
])

[I,M,CT,t] = f(16,5.2,14,730,0.11,0.5,10,70,0.007)
graph_sim = {"data": [{"x":t, "y":I, "name":"Infectados", "showlegend":True}, {"x":t, "y":M, "name":"Mortos", "showlegend":True}, {"x":t, "y":0.015*CT, "name":"Casos Críticos Acumulados", "showlegend":True}], "layout":{"width":800, "height":500, "margin":{"l":30, "r":0, "t":20,"b":30}}}

layout_simulation = html.Div(style={"width":"100vw", "margin-left":-8, "margin-top":-10}, children=[
	html.Div(style={"height":70, "width":"100%", "background-color":"#F29C04", "display":"flex"}, children=[
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":238, "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.Img(src="/assets/click_covid.jpg", style={"height":70, "margin":{"l":0, "r":0, "t":0, "b":0}})], href="/")
			#~ dcc.Link([html.H2("Click Covid")], href="/", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"10%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Cidades")], href="/page-cities", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"}, children=[
			html.A([html.H3("Simulação")], href="/page-simulation", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"}, children=[
			html.A([html.H3("Análise Socioeconômica")], href="/page-socio", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Equipe")], href="/page-equipe", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			dcc.Link([html.H3("Agradecimentos")], href="/page-thanks", className="bar"),
		]),
		html.Div(style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":70, "width":"15%", "text-align":"center", "vertical-align":"middle"},children=[
			html.A([html.H3("GitHub")], href="https://github.com/click-covid/clickcovid-heroku", target="_blank", className="bar"),
		]),
	]),
	
	html.Div(style={"height":20}),
	
	html.Div(style={"display":"flex"}, children=[
		html.Div(style={}, children=[
			dcc.Graph(figure=graph_sim, id="graph-sim"),
		]),
		html.Div(style={"width":50}),
		html.Div(style={"text-align":"center"}, children=[
			html.Div(style={"width":400}, children=[
				html.P(id='slider-output-container-tr'),    
				dcc.Slider(
					id='my-slider-tr',
					min=1,
					max=20,
					step=0.01,
					value=16,
				),
			]),
                
			html.Div(style={"width":400}, children=[
				html.P(id='slider-output-container-c'),    
				dcc.Slider(
					id='my-slider-c',
					min=0.5,
					max=15,
					step=0.1,
					value=5.2,
				),
			]),


			html.Div(style={"width":400}, children=[
				html.P(id='slider-output-container-td'),    
				dcc.Slider(
					id='my-slider-td',
					min=1,
					max=20,
					step=0.01,
					value=14,
				),
			]),


			html.Div(style={"width":400}, children=[
				html.P(id='slider-output-container-re'),    
				dcc.Slider(
					id='my-slider-re',
					min=30,
					max=1000,
					step=1,
					value=730,
				),
			]),


			html.Div(style={"width":400}, children=[
				html.P(id='slider-output-container-Pb'),    
				dcc.Slider(
					id='my-slider-Pb',
					min=0,
					max=1,
					step=0.01,
					value=0.11,
				),
			]),
                        
            
            html.Div(style={"width":400}, children=[
				html.P(id='slider-output-container-Ef'),    
				dcc.Slider(
					id='my-slider-Ef',
					min=0,
					max=1,
					step=0.01,
					value=0.5,
				),
			]),
            
            
            html.Div(style={"width":400}, children=[
				html.P(id='slider-output-container-tc'),    
				dcc.Slider(
					id='my-slider-tc',
					min=0,
					max=1000,
					step=1,
					value=10,
				),
			]),
            
            
            html.Div(style={"width":400}, children=[
				html.P(id='slider-output-container-tf'),    
				dcc.Slider(
					id='my-slider-tf',
					min=0,
					max=1000,
					step=1,
					value=60,
				),
			]),
            
            
            html.Div(style={"width":400}, children=[
				html.P(id='slider-output-container-ifr'),    
				dcc.Slider(
					id='my-slider-ifr',
					min=0,
					max=0.2,
					step=0.001,
					value=0.007,
				),
			]),
		]),
	])
])

url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=True),
    html.Div(id='page-content')
])

app.layout = url_bar_and_content_div

app.validation_layout = html.Div([
	url_bar_and_content_div,
    layout_home,
	layout_cities,
	layout_socio,
	layout_equipe,
	layout_thanks,
	layout_simulation
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
	if pathname == "/page-cities":
		return layout_cities
	elif pathname == "/page-equipe":
		return layout_equipe
	elif pathname == "/page-thanks":
		return layout_thanks
	elif pathname == "/page-socio":
		return layout_socio
	elif pathname == "/page-simulation":
		return layout_simulation
	else:
		return layout_home

@app.callback(
    [Output('slider-output-container-tr', 'children'), Output('slider-output-container-c', 'children'), Output('slider-output-container-td', 'children'), Output('slider-output-container-re', 'children'), Output('slider-output-container-Pb', 'children'), Output('slider-output-container-Ef', 'children'), Output('slider-output-container-tc', 'children'), Output('slider-output-container-tf', 'children'), Output('slider-output-container-ifr', 'children'), Output('graph-sim', 'figure')],
    [Input('my-slider-tr', 'value'),
     Input('my-slider-c', 'value'),
     Input('my-slider-td', 'value'),
     Input('my-slider-re', 'value'),
     Input('my-slider-Pb', 'value'),
     Input('my-slider-Ef', 'value'),
     Input('my-slider-tc', 'value'),
     Input('my-slider-tf', 'value'),
     Input('my-slider-ifr', 'value')])
def update_func_tr(value_1,value_2,value_3,value_4,value_5,value_6,value_7,value_8,value_9):

    tr = value_1
    c = value_2
    td = value_3
    re = value_4
    Pb = value_5
    Ef = value_6
    tc = value_7
    tf = value_8
    ifr = value_9
    [I,M,CT,t] = f(tr,c,td,re,Pb,Ef,tc,tf,ifr)
    

    return 'Tempo do início dos sintomas até recuperação: {:.4f}'.format(value_1), 'Tempo de Incubação: {:.4f}'.format(value_2), 'Tempo do início dos sintomas até morte: {:.4f}'.format(value_3), 'Tempo médio até a possibilidade de reinfecção: {:.4f}'.format(value_4), 'Probabilidade de Infecção: {:.4f}'.format(value_5), 'Eficiência da intervenção social: {:.4f}'.format(value_6), 'Dia de início da intervenção: {:.4f}'.format(value_7), 'Dia final da intervenção: {:.4f}'.format(value_8), 'Mortalidade por infecção: {:.4f}'.format(value_9),{"data": [{"x":t, "y":I, "name":"Infectados", "showlegend":True}, {"x":t, "y":M, "name":"Mortos", "showlegend":True}, {"x":t, "y":0.015*CT, "name":"Casos Críticos Acumulados", "showlegend":True}], "layout":{"width":800, "height":500, "margin":{"l":30, "r":0, "t":20,"b":30}}},

@app.callback(
    [Output('drop-cities', 'options'), Output('graph-cities', 'figure')],
    [Input('drop-states2', 'value'), Input('drop-cities', 'value')])
def update_output(st, ct):
	
	st_ = ct_dat[ct_dat.state==st]
	stcts_ = st_.name.unique()
	
	if(ct == 0):
		
		return [{"label":"Todo o Estado", "value":0}]+[{'label': i, 'value': i} for i in stcts_], {"data": [{"x":st_dat[st_dat.state==st].date, "y":st_dat[st_dat.state==st].cases, "name":"Casos", "showlegend":True}, {"x":st_dat[st_dat.state==st].date, "y":st_dat[st_dat.state==st].deaths, "name":"Mortes", "showlegend":True}], "layout":{"width":"100%", "margin":{"l":30, "r":0, "t":20, "b":30}}}
	
	else:
		
		return [{"label":"Todo o Estado", "value":0}]+[{'label': i, 'value': i} for i in stcts_], {"data": [{"x":st_[st_.name==ct].date, "y":st_[st_.name==ct].cases, "name":"Casos", "showlegend":True}, {"x":st_[st_.name==ct].date, "y":st_[st_.name==ct].deaths, "name":"Mortes", "showlegend":True}], "layout":{"width":"100%", "margin":{"l":30, "r":0, "t":20, "b":30}}}

if __name__ == '__main__':
    app.run_server(debug=True)
