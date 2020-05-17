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
from scipy.integrate import odeint

################################################
 ## SET UP MODEL EQUATIONS

########### Define your variables
beers=['Chesapeake Stout', 'Snake Dog IPA', 'Imperial Porter', 'Double Dog IPA']
ibu_values=[35, 60, 85, 75]
abv_values=[5.4, 7.1, 9.2, 4.3]
color1='lightblue'
color2='darkgreen'
mytitle='Beer Comparison'
tabtitle='Click COVID-19'
myheading='Click COVID-19: uma ferramenta interativa para o estudo da pandemia'
label1='IBU'
label2='ABV'
githublink='https://github.com/YokoSenju/flying-dog-beers'
sourceurl='https://www.flyingdog.com/beers/'

########### Set up the chart

def f(a,c,mu,d,Pb):
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
            
            
                #a = 0.071
                #c = 1/5.2
                #mu = 0.053
                #d = 0.00137
                #Pb = 0.11
                Pk = Pb*0.08
                Ej = np.array([E1,E2,E3,E4,E5,E6,E7,E8])
                Ij = np.array([I1,I2,I3,I4,I5,I6,I7,I8])
                B1j = sum(Cij[0,:]*Ij)
                k1j = sum(Cij[0,:]*Ej)
            
                B1 = (B1j*Pb/N1)
                c1 = c
                a1 = 10*a
                mu1 = 0
                k1 = k1j*Pk/N1 
                dS1dt = -B1*S1 - k1*S1 + d*R1
                dE1dt = B1*S1 - c1*E1 + k1*S1
                dI1dt = c1*E1 - a1*I1 - mu1*I1
                dR1dt = a1*I1 - d*R1
                dM1dt = mu1*I1
            
            
                B2j = sum(Cij[1,:]*Ij)
                k2j = sum(Cij[1,:]*Ej)
            
                B2 = (B2j*Pb/N2)
                c2 = c
                a2 = (1-0.002)*a
                mu2 = 0.002*mu
                k2 = k2j*Pk/N2
                dS2dt = -B2*S2 - k2*S2 + d*R2
                dE2dt = B2*S2 - c2*E2 + k2*S2
                dI2dt = c2*E2 - a2*I2 - mu2*I2
                dR2dt = a2*I2 - d*R2
                dM2dt = mu2*I2
            
            
                B3j = sum(Cij[2,:]*Ij)
                k3j = sum(Cij[2,:]*Ej)
            
                B3 = (B3j*Pb/N3)
                c3 = c
                a3 = (1-0.002)*a
                mu3 = 0.002*mu
                k3 = k3j*Pk/N3
                dS3dt = -B3*S3 - k3*S3 + d*R3
                dE3dt = B3*S3 - c3*E3 + k3*S3
                dI3dt = c3*E3 - a3*I3 - mu3*I3
                dR3dt = a3*I3 - d*R3
                dM3dt = mu3*I3
                
                B4j = sum(Cij[3,:]*Ij)
                k4j = sum(Cij[3,:]*Ej)
            
                B4 = (B4j*Pb/N4)
                c4 = c
                a4 = (1-0.002)*a
                mu4 = 0.002*mu
                k4 = k4j*Pk/N4 
                dS4dt = -B4*S4 - k4*S4 + d*R4
                dE4dt = B4*S4 - c4*E4 + k4*S4
                dI4dt = c4*E4 - a4*I4 - mu4*I4
                dR4dt = a4*I4 - d*R4
                dM4dt = mu4*I4
            
            
                B5j = sum(Cij[4,:]*Ij)
                k5j = sum(Cij[4,:]*Ej)
            
                B5 = (B5j*Pb/N5)
                c5 = c
                a5 = (1-0.004)*a
                mu5 = 0.004*mu
                k5 = k5j*Pk/N5 
                dS5dt = -B5*S5 - k5*S5 + d*R5
                dE5dt = B5*S5 - c5*E5 + k5*S5
                dI5dt = c5*E5 - a5*I5 - mu5*I5
                dR5dt = a5*I5 - d*R5
                dM5dt = mu5*I5
            
            
                B6j = sum(Cij[5,:]*Ij)
                k6j = sum(Cij[5,:]*Ej)
            
                B6 = (B6j*Pb/N6)
                c6 = c
                a6 = (1-0.013)*a
                mu6 = 0.013*mu
                k6 = k6j*Pk/N6 
                dS6dt = -B6*S6 - k6*S6 + d*R6
                dE6dt = B6*S6 - c6*E6 + k6*S6
                dI6dt = c6*E6 - a6*I6 - mu6*I6
                dR6dt = a6*I6 - d*R6
                dM6dt = mu6*I6
            
            
                B7j = sum(Cij[6,:]*Ij)
                k7j = sum(Cij[6,:]*Ej)
            
                B7 = (B7j*Pb/N7)
                c7 = c
                a7 = (1-0.036)*a
                mu7 = 0.036*mu
                k7 = k7j*Pk/N7 
                dS7dt = -B7*S7 - k7*S7 + d*R7
                dE7dt = B7*S7 - c7*E7 + k7*S7
                dI7dt = c7*E7 - a7*I7 - mu7*I7
                dR7dt = a7*I7 - d*R7
                dM7dt = mu7*I7
            
            
                B8j = sum(Cij[7,:]*Ij)
                k8j = sum(Cij[7,:]*Ej)
            
                B8 = (B8j*Pb/N8)
                c8 = c
                a8 = (1-0.114)*a
                mu8 = 0.114*mu
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
        t = np.linspace(1, 730, 1000)
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




################################################

st_dat = pd.read_csv(r"https://storage.googleapis.com/kagglesdsdata/datasets/549702/1162499/brazil_covid19.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1589944851&Signature=RWmeNQ%2BIQzTrQblJX5C6o9TF1tQcNCvlTwz5CbSqsSVaMqbB2eBtpOLwFlZ639M54OHybmTEsS%2FQVPa2Yk9JulE%2FbmwRStKcZD2lyGqsM4OKoL%2FDEUtjTA9VAL%2BcBrYTDaULIYoSnsEUx4FPoj3hxlAA6n%2Fbn4hFUgqJ38JErFcYLX7eQIXs4gI0ddeeZ86npcVdtJKcpSidL4dIjA8uEKhtyKt%2Fe7l%2F7tgpGs7kiHoTfjVeD8f35Q1qjnFmfEMeD2l2T%2FCyEBzxLzp%2BQkeQvoXrik3LeuCMk18%2FLFklPAuTc7XCeSqRCKf0kOCkCiJPIfgrN7NukdAJOXRcgb3ijQ%3D%3D&response-content-disposition=attachment%3B+filename%3Dbrazil_covid19.csv")

url_ct_dat = requests.get(r"https://storage.googleapis.com/kaggle-data-sets/549702/1162499/compressed/brazil_covid19_cities.csv.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1589944913&Signature=Bdp%2FbEvGty6o672tmkAOEzE1%2BXr30nkniPr7pGbrpnMCcTK8yuB2vlQe18uhsYc5lceiNpu1IKCYwW2KiSljmqKbvgP97qEqxcbJdf2yzApieKPLxcChxElyCmFl5oqReFq6YTHgTHRjv%2F%2BiFn3mGG7Na8oINswPTl5yGm5ZDtroewW7yT3ttyhbkCooQxWJ40lBRqPdgsv3GW4kwYN1u5ePlfso5pS8vZJnCRHAcLpfrk7MBbLL2k%2FdgAUfZeLmlGVApeG7oKFZl1aRGT5%2B82JyqrmL2VZA7IjINnAbjBBqK1JHgQvhnArOTTIS6JuXdU8aoOdxiatTi8Tsp4D02g%3D%3D&response-content-disposition=attachment%3B+filename%3Dbrazil_covid19_cities.csv.zip")

zipfile = ZipFile(BytesIO(url_ct_dat.content))

ct_dat = pd.read_csv(zipfile.open(zipfile.namelist()[0]))

states = np.sort(st_dat.state.unique())

x1 = np.linspace(0, 10, 1000)

app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True
server = app.server
app.title = "CORONAVAIRUS"

layout_home = html.Div([
	html.Div(
		style={"height":100, "width":"100vw", "margin-left":-8, "margin-top":-10, "background-color":"#9015BD", "display":"flex"},
		children=[
			html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					dcc.Link([html.H1("Click Covid")], href="/", style={"color":"black", "text-decoration":"none"}),
				],
			),
			#~ html.Div(
				#~ style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				#~ children=[
					#~ dcc.Link([html.H1("States")], href="/page-states", style={"color":"black", "text-decoration":"none"}),
				#~ ],
			#~ ),
			html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					dcc.Link([html.H1("Cities")], href="/page-cities", style={"color":"black", "text-decoration":"none"}),
				],
			),
			html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					dcc.Link([html.H1("Equipe")], href="/page-equipe", style={"color":"black", "text-decoration":"none"}),
				],
			),
                        html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					html.A([html.H1("Simulation")], href="/page-simulation", style={"color":"black", "text-decoration":"none"}),
				],
			),
			html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					html.A([html.H1("GitHub")], href="https://github.com/joaopedro-vm/flying-dog-beers", target="_blank", style={"color":"black", "text-decoration":"none"}),
				],
			),
		],
	),
	html.Img(src="/assets/corona_cinza_fundo_Prancheta_1.jpg", style={"margin-left":-8})
])

graph_states = {"data": [{"x":st_dat[st_dat.state==states[0]].date, "y":st_dat[st_dat.state==states[0]].cases, "name":"Casos", "showlegend":True}, {"x":st_dat[st_dat.state==states[0]].date, "y":st_dat[st_dat.state==states[0]].deaths, "name":"Mortes", "showlegend":True}], "layout":{"width":500, "height":300, "margin":{"l":40, "r":0, "t":20, "b":30}}}

#~ layout_states = html.Div([
	#~ html.Div(
		#~ style={"height":100, "width":"100vw", "margin-left":-8, "margin-top":-10, "background-color":"#9015BD", "display":"flex"},
		#~ children=[
			#~ html.Div(
				#~ style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				#~ children=[
					#~ dcc.Link([html.H1("Click Covid")], href="/", style={"color":"black", "text-decoration":"none"}),
				#~ ],
			#~ ),
			#~ html.Div(
				#~ style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				#~ children=[
					#~ dcc.Link([html.H1("States")], href="/page-states", style={"color":"black", "text-decoration":"none"}),
				#~ ],
			#~ ),
			#~ html.Div(
				#~ style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				#~ children=[
					#~ dcc.Link([html.H1("Cities")], href="/page-cities", style={"color":"black", "text-decoration":"none"}),
				#~ ],
			#~ ),
			#~ html.Div(
				#~ style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				#~ children=[
					#~ html.A([html.H1("GitHub")], href="https://github.com/joaopedro-vm/flying-dog-beers", target="_blank", style={"color":"black", "text-decoration":"none"}),
				#~ ],
			#~ ),
		#~ ],
	#~ ),
	#~ html.Div(style={"height":20}),
	#~ html.Div(
	#~ [
		#~ html.P(id='drop-out-states', children=['Estado: '], style={"width":60}),    
		#~ dcc.Dropdown(
			#~ id="drop-states",
			#~ options=[{'label': i, 'value': i} for i in states],
			#~ value=states[0],
			#~ style={"width":200}
		#~ ),
	#~ ],
	#~ style={"width":300, "display":"flex"}
	#~ ),
	#~ dcc.Graph(id='graph-states', figure=graph_states),
	#~ html.P(["Fonte: ", html.A(["kaggle/unanimad/corona-virus-brazil"], href="https://www.kaggle.com/unanimad/corona-virus-brazil", target="_blank")]),
#~ ])

st = ct_dat[ct_dat.state==states[0]]
stcts = st.name.unique()

graph_cities = {"data": [{"x":st[st.name==stcts[0]].date, "y":st[st.name==stcts[0]].cases, "name":"Casos", "showlegend":True}, {"x":st[st.name==stcts[0]].date, "y":st[st.name==stcts[0]].deaths, "name":"Mortes", "showlegend":True}], "layout":{"width":500, "height":300, "margin":{"l":40, "r":0, "t":20, "b":30}}}

layout_cities = html.Div([
	html.Div(
		style={"height":100, "width":"100vw", "margin-left":-8, "margin-top":-10, "background-color":"#9015BD", "display":"flex"},
		children=[
			html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					dcc.Link([html.H1("Click Covid")], href="/", style={"color":"black", "text-decoration":"none"}),
				],
			),
			#~ html.Div(
				#~ style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				#~ children=[
					#~ dcc.Link([html.H1("States")], href="/page-states", style={"color":"black", "text-decoration":"none"}),
				#~ ],
			#~ ),
			html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					dcc.Link([html.H1("Cities")], href="/page-cities", style={"color":"black", "text-decoration":"none"}),
				],
			),
			html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					dcc.Link([html.H1("Equipe")], href="/page-equipe", style={"color":"black", "text-decoration":"none"}),
				],
			),
                        html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					html.A([html.H1("Simulation")], href="/page-simulation", style={"color":"black", "text-decoration":"none"}),
				],
			),
			html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					html.A([html.H1("GitHub")], href="https://github.com/joaopedro-vm/flying-dog-beers", target="_blank", style={"color":"black", "text-decoration":"none"}),
				],
			),
		],
	),
	html.Div(style={"height":20,}),
	html.Div(
	[
		html.P(id='drop-out-states2', children=['Estado:'], style={"width":60}),    
		dcc.Dropdown(
			id="drop-states2",
			options=[{'label': i, 'value': i} for i in states],
			value="São Paulo",
			style={"width":200}
		),
	],
	style={"width":300, "display":"flex"}
	),
	html.Div(
	[
		html.P(id='drop-out-cities', children=['Cidade:'], style={"width":60}),    
		dcc.Dropdown(
			id="drop-cities",
			options=[{"label":"Todo o Estado", "value":0}] + [{'label': i, 'value': i} for i in stcts],
			value=0,
			style={"width":200}
		),
	],
	style={"width":300, "display":"flex"}
	),
	dcc.Graph(id='graph-cities', figure=graph_states),
	html.P(["Fonte: ", html.A(["kaggle/unanimad/corona-virus-brazil"], href="https://www.kaggle.com/unanimad/corona-virus-brazil", target="_blank")]),
])

layout_equipe = html.Div([
	html.Div(
		style={"height":100, "width":"100vw", "margin-left":-8, "margin-top":-10, "background-color":"#9015BD", "display":"flex"},
		children=[
			html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					dcc.Link([html.H1("Click Covid")], href="/", style={"color":"black", "text-decoration":"none"}),
				],
			),
			#~ html.Div(
				#~ style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				#~ children=[
					#~ dcc.Link([html.H1("States")], href="/page-states", style={"color":"black", "text-decoration":"none"}),
				#~ ],
			#~ ),
			html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					dcc.Link([html.H1("Cities")], href="/page-cities", style={"color":"black", "text-decoration":"none"}),
				],
			),
			html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					dcc.Link([html.H1("Equipe")], href="/page-equipe", style={"color":"black", "text-decoration":"none"}),
				],
			),
                        html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					html.A([html.H1("Simulation")], href="/page-simulation", style={"color":"black", "text-decoration":"none"}),
				],
			),
			html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					html.A([html.H1("GitHub")], href="https://github.com/joaopedro-vm/flying-dog-beers", target="_blank", style={"color":"black", "text-decoration":"none"}),
				],
			),
		],
	),
	html.Div(style={"height":20}),
	html.Div(style={"height":300, "width":"100vw", "margin-left":-8, "display":"flex"}, children=[
		html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":300, "width":400, "text-align":"center"},
				children=[
					html.Img(src="/assets/covid_bolinha.png", style={"width":200, "height":200}),
					html.P(["Corona 1"]),
					html.P(["Estudante da Universidade dos Coronas"])
				],
			),
		html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":300, "width":400, "text-align":"center"},
				children=[
					html.Img(src="/assets/covid_bolinha.png", style={"width":200, "height":200}),
					html.P(["Corona 2"]),
					html.P(["Estudante da Universidade dos Coronas"])
				],
			),
		html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":300, "width":400, "text-align":"center"},
				children=[
					html.Img(src="/assets/covid_bolinha.png", style={"width":200, "height":200}),
					html.P(["Corona 3"]),
					html.P(["Estudante da Universidade dos Coronas"])
				],
			),
		html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "height":300, "width":400, "text-align":"center"},
				children=[
					html.Img(src="/assets/covid_bolinha.png", style={"width":200, "height":200}),
					html.P(["Corona 4"]),
					html.P(["Estudante da Universidade dos Coronas"])
				],
			),
	]
	),
])


layout_simulation = html.Div([
	html.Div(
		style={"height":100, "width":"100vw", "margin-left":-8, "margin-top":-10, "background-color":"#9015BD", "display":"flex"},
		children=[
			html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					dcc.Link([html.H1("Click Covid")], href="/", style={"color":"black", "text-decoration":"none"}),
				],
			),
			#~ html.Div(
				#~ style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				#~ children=[
					#~ dcc.Link([html.H1("States")], href="/page-states", style={"color":"black", "text-decoration":"none"}),
				#~ ],
			#~ ),
			html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					dcc.Link([html.H1("Cities")], href="/page-cities", style={"color":"black", "text-decoration":"none"}),
				],
			),
			html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					dcc.Link([html.H1("Equipe")], href="/page-equipe", style={"color":"black", "text-decoration":"none"}),
				],
			),
			html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					html.A([html.H1("Simulation")], href="/page-simulation", style={"color":"black", "text-decoration":"none"}),
				],
			),
                        html.Div(
				style={"margin":{"l":0, "r":0, "t":0, "b":0}, "padding":10, "height":100, "width":300, "text-align":"center"},
				children=[
					html.A([html.H1("GitHub")], href="https://github.com/joaopedro-vm/flying-dog-beers", target="_blank", style={"color":"black", "text-decoration":"none"}),
				],
			),
		],
	),
################################################
	html.Div(style={"height":20,}),
        html.Div(children=[
                html.H2(myheading),
                 dcc.Graph(
                 id='flyingdog',
                 figure=beer_fig
                )]),
                html.Div([
                            html.P(id='slider-output-container-a'),    
                            dcc.Slider(
                                id='my-slider-a',
                                min=0.03,
                                max=0.2,
                                step=0.0017,
                                value=0.071,
                        ),
                    ],
                    style={"width":500}
                    ),
                
                html.Div([
                            html.P(id='slider-output-container-c'),    
                            dcc.Slider(
                                id='my-slider-c',
                                min=0.06,
                                max=1,
                                step=0.0094,
                                value=1/5.2,
                        ),
                    ],
                    style={"width":500}
                    ),


                html.Div([
                            html.P(id='slider-output-container-mi'),    
                            dcc.Slider(
                                id='my-slider-mi',
                                min=0.03,
                                max=0.2,
                                step=0.0017,
                                value=0.053,
                        ),
                    ],
                    style={"width":500}
                    ),


                html.Div([
                            html.P(id='slider-output-container-d'),    
                            dcc.Slider(
                                id='my-slider-d',
                                min=0.000913,
                                max=0.0027,
                                step=0.00001787,
                                value=0.00137,
                        ),
                    ],
                    style={"width":500}
                    ),


                html.Div([
                            html.P(id='slider-output-container-Pb'),    
                            dcc.Slider(
                                id='my-slider-Pb',
                                min=0,
                                max=1,
                                step=0.01,
                                value=0.11,
                        ),
                    ],
                    style={"width":500}
                    ),
                
                html.A('Github', href=githublink),
                html.Br(),
                html.A('Source', href=sourceurl),
                ]
            )
         

######################################################

url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=True),
    html.Div(id='page-content')
])

app.layout = url_bar_and_content_div

app.validation_layout = html.Div([
	url_bar_and_content_div,
    layout_home,
    #~ layout_states,
	layout_cities,
	layout_equipe
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
	if pathname == "/page-cities":
		return layout_cities
	elif pathname == "/page-equipe":
		return layout_equipe
	#~ elif pathname == "/page-states":
		#~ return layout_states
	elif pathname == "/page-simulation":
		return layout_simulation
	else:
		return layout_home

#~ @app.callback(
    #~ Output('graph-states', 'figure'),
    #~ [Input('drop-states', 'value')])
#~ def update_output(value):

	#~ return {"data": [{"x":st_dat[st_dat.state==value].date, "y":st_dat[st_dat.state==value].cases, "name":"Casos", "showlegend":True}, {"x":st_dat[st_dat.state==value].date, "y":st_dat[st_dat.state==value].deaths, "name":"Mortes", "showlegend":True}], "layout":{"width":500, "height":300, "margin":{"l":40, "r":0, "t":20, "b":30}}}

@app.callback(
    [Output('drop-cities', 'options'), Output('graph-cities', 'figure')],
    [Input('drop-states2', 'value'), Input('drop-cities', 'value')])
def update_output(st, ct):
	
	st_ = ct_dat[ct_dat.state==st]
	stcts_ = st_.name.unique()
	
	if(ct == 0):
		
		return [{"label":"Todo o Estado", "value":0}]+[{'label': i, 'value': i} for i in stcts_], {"data": [{"x":st_dat[st_dat.state==st].date, "y":st_dat[st_dat.state==st].cases, "name":"Casos", "showlegend":True}, {"x":st_dat[st_dat.state==st].date, "y":st_dat[st_dat.state==st].deaths, "name":"Mortes", "showlegend":True}], "layout":{"width":500, "height":300, "margin":{"l":40, "r":0, "t":20, "b":30}}}
	
	else:
		
		return [{"label":"Todo o Estado", "value":0}]+[{'label': i, 'value': i} for i in stcts_], {"data": [{"x":st_[st_.name==ct].date, "y":st_[st_.name==ct].cases, "name":"Casos", "showlegend":True}, {"x":st_[st_.name==ct].date, "y":st_[st_.name==ct].deaths, "name":"Mortes", "showlegend":True}], "layout":{"width":500, "height":300, "margin":{"l":40, "r":0, "t":20, "b":30}}}


@app.callback(
    [Output('slider-output-container-a', 'children'), Output('flyingdog', 'figure')],
    [Input('my-slider-a', 'value'),
     Input('my-slider-c', 'value'),
     Input('my-slider-mi', 'value'),
     Input('my-slider-d', 'value'),
     Input('my-slider-Pb', 'value')])

def update_func_a(value_1,value_2,value_3,value_4,value_5):

    a = value_1
    c = value_2
    mi = value_3
    d = value_4
    Pb = value_5
    [I,M,CT,t] = f(a,c,mi,d,Pb)
    

    return  (html.P('Parâmetros '),html.P('Taxa de recuperação: {}'.format(value_1)),
            html.P('Tempo de Incubação: {}'.format(value_2)),
            html.P('Probabilidade de Morte: {}'.format(value_3)),
            html.P('Taxa de Reinfecção: {}'.format(value_4)),
            html.P('Probabilidade de Infecção: {}'.format(value_5))),{
            "data": [{"x":t,
                      "y":I,
                      "name":"Infectados",
                      "showlegend":True},
                     {"x":t,
                      "y":M,
                      "name":"Mortos",
                      "showlegend":True},
                     {"x":t,
                      "y":0.05*CT,
                      "name":"Casos Acumulados",
                      "showlegend":True}]},
            

 
if __name__ == '__main__':
    app.run_server(debug=True)
