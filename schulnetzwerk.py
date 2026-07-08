import streamlit as st
import pandas as pd
import graphviz
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator

st.set_page_config(page_title="Advanced Schul-KI", layout="wide")

# -----------------------------------------------------------------------------
# 1. MODELL-STRUKTUR & BASIS-WERTE
# -----------------------------------------------------------------------------
@st.cache_resource
def get_initial_model():
    model = DiscreteBayesianNetwork([
        ('Vorkenntnisse', 'Verstaendnis'),
        ('Hausaufgaben', 'Verstaendnis'),
        ('Mitarbeit', 'Verstaendnis'),
        ('Fehlzeiten', 'Verstaendnis'),      
        ('Fehlzeiten', 'Pruefungsangst'),    
        ('Verstaendnis', 'Pruefung_bestanden'),
        ('Pruefungsangst', 'Pruefung_bestanden')
    ])
    return model

# -----------------------------------------------------------------------------
# 2. SEITENLEISTE & NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("🛠️ Modell-Konfiguration")
app_mode = st.sidebar.selectbox("Modus wählen", ["Interaktive Vorhersage", "Modell mit Daten trainieren"])

# -----------------------------------------------------------------------------
# 3. MODUS: TRAINING AUS CSV
# -----------------------------------------------------------------------------
if app_mode == "Modell mit Daten trainieren":
    st.header("📂 Modell-Training via CSV")
    st.info("Laden Sie eine CSV-Datei hoch. Die Spalten müssen exakt den Knotennamen entsprechen (0=Schlecht, 1=Gut).")
    
    uploaded_file = st.file_uploader("Wähle eine CSV Datei", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Vorschau der Daten:")
        st.dataframe(data.head())
        
        if st.button("Training starten"):
            model = get_initial_model()
            # Training mit Maximum Likelihood
            mle = MaximumLikelihoodEstimator(model, data)
            
            try:
                cpds = mle.get_parameters()
                # Speichere die trainierten CPDs in der session_state
                st.session_state['trained_cpds'] = {cpd.variable: cpd for cpd in cpds}
                st.success("Erfolg! Das Modell wurde mit Ihren Daten trainiert.")
            except Exception as e:
                st.error(f"Fehler beim Training: {e}")

# -----------------------------------------------------------------------------
# 4. MODUS: VORHERSAGE & MANUELLE ANPASSUNG
# -----------------------------------------------------------------------------
else:
    st.header("📊 Interaktive Vorhersage & Experten-Input")
    
    # Tabs für bessere Übersicht
    tab_pred, tab_expert = st.tabs(["🎯 Vorhersage", "⚙️ Wahrscheinlichkeiten (CPDs) anpassen"])
    
    model = get_initial_model()
    
    with tab_expert:
        st.subheader("Manuelle Anpassung der Basis-Wahrscheinlichkeiten")
        st.write("Passen Sie hier die 'A-priori'-Werte an (Wahrscheinlichkeit für 'Gut' bzw. 'Ja').")
        
        # Falls trainiert wurde, nimm die trainierten Werte als Startpunkt
        def get_default_val(var, state_idx=1):
            if 'trained_cpds' in st.session_state:
                return float(st.session_state['trained_cpds'][var].values[state_idx])
            return 0.7 # Default fallback

        v_val = st.slider("P(Vorkenntnisse = Gut)", 0.0, 1.0, get_default_val('Vorkenntnisse'))
        h_val = st.slider("P(Hausaufgaben = Gut)", 0.0, 1.0, get_default_val('Hausaufgaben'))
        m_val = st.slider("P(Mitarbeit = Gut)", 0.0, 1.0, get_default_val('Mitarbeit'))
        f_val = st.slider("P(Hohe Fehlzeiten)", 0.0, 1.0, 0.15) # Default 15%

        # CPDs bauen (Manuell basierend auf Slidern)
        cpd_v = TabularCPD('Vorkenntnisse', 2, [[1-v_val], [v_val]])
        cpd_h = TabularCPD('Hausaufgaben', 2, [[1-h_val], [h_val]])
        cpd_m = TabularCPD('Mitarbeit', 2, [[1-m_val], [m_val]])
        cpd_f = TabularCPD('Fehlzeiten', 2, [[1-f_val], [f_val]])
        
        # Für komplexe abhängige Knoten nutzen wir hier die Original-Logik oder trainierte Werte
        # (Zur Vereinfachung nehmen wir hier die trainierten oder die Standard-Tabellen)
        if 'trained_cpds' in st.session_state:
            cpd_angst = st.session_state['trained_cpds']['Pruefungsangst']
            cpd_verst = st.session_state['trained_cpds']['Verstaendnis']
            cpd_pruefung = st.session_state['trained_cpds']['Pruefung_bestanden']
        else:
            # Fallback auf Standardwerte aus dem vorigen Prompt
            cpd_angst = TabularCPD('Pruefungsangst', 2, [[0.85, 0.40], [0.15, 0.60]], evidence=['Fehlzeiten'], evidence_card=[2])
            cpd_verst = TabularCPD('Verstaendnis', 2, [[0.95, 0.99, 0.70, 0.85, 0.60, 0.80, 0.30, 0.50, 0.50, 0.75, 0.25, 0.45, 0.15, 0.35, 0.05, 0.15], [0.05, 0.01, 0.30, 0.15, 0.40, 0.20, 0.70, 0.50, 0.50, 0.25, 0.75, 0.55, 0.85, 0.65, 0.95, 0.85]], evidence=['Vorkenntnisse', 'Hausaufgaben', 'Mitarbeit', 'Fehlzeiten'], evidence_card=[2, 2, 2, 2])
            cpd_pruefung = TabularCPD('Pruefung_bestanden', 2, [[0.85, 0.95, 0.05, 0.25], [0.15, 0.05, 0.95, 0.75]], evidence=['Verstaendnis', 'Pruefungsangst'], evidence_card=[2, 2])

        model.add_cpds(cpd_v, cpd_h, cpd_m, cpd_f, cpd_angst, cpd_verst, cpd_pruefung)
        inference = VariableElimination(model)

    with tab_pred:
        # Hier kommt die Logik aus der vorigen App hin (Evidenz wählen)
        st.subheader("Aktuelles Schülerprofil")
        evid = {}
        c1, c2 = st.columns(2)
        with c1:
            v_ev = st.selectbox("Vorkenntnisse", ["Unbekannt", "Schlecht", "Gut"])
            if v_ev != "Unbekannt": evid['Vorkenntnisse'] = 1 if v_ev == "Gut" else 0
            
            f_ev = st.selectbox("Fehlzeiten", ["Unbekannt", "Normal", "Viele"])
            if f_ev != "Unbekannt": evid['Fehlzeiten'] = 1 if f_ev == "Viele" else 0
        
        with c2:
            res = inference.query(variables=['Pruefung_bestanden'], evidence=evid)
            prob = res.values[1] * 100
            st.metric("Wahrscheinlichkeit Bestehen", f"{prob:.1f}%")
            st.progress(int(prob))
            
            # Graph zeichnen
            dot = graphviz.Digraph()
            dot.edge('Vorkenntnisse', 'Verstaendnis')
            dot.edge('Hausaufgaben', 'Verstaendnis')
            dot.edge('Mitarbeit', 'Verstaendnis')
            dot.edge('Fehlzeiten', 'Verstaendnis')
            dot.edge('Fehlzeiten', 'Pruefungsangst')
            dot.edge('Verstaendnis', 'Pruefung_bestanden')
            dot.edge('Pruefungsangst', 'Pruefung_bestanden')
            st.graphviz_chart(dot)
