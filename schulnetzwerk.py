import streamlit as st
import graphviz
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# -----------------------------------------------------------------------------
# 1. INITIALISIERUNG DES ORIGINALEN NETZWERKS
# -----------------------------------------------------------------------------
@st.cache_resource
def init_custom_network():
    model = DiscreteBayesianNetwork([
        ('Vorkenntnisse', 'Verstaendnis'),
        ('Hausaufgaben', 'Verstaendnis'),
        ('Mitarbeit', 'Verstaendnis'),
        ('Fehlzeiten', 'Verstaendnis'),      
        ('Fehlzeiten', 'Pruefungsangst'),    
        ('Verstaendnis', 'Pruefung_bestanden'),
        ('Pruefungsangst', 'Pruefung_bestanden')
    ])

    cpd_vorkenntnisse = TabularCPD(variable='Vorkenntnisse', variable_card=2, values=[[0.3], [0.7]])
    cpd_hausaufgaben = TabularCPD(variable='Hausaufgaben', variable_card=2, values=[[0.25], [0.75]])
    cpd_mitarbeit = TabularCPD(variable='Mitarbeit', variable_card=2, values=[[0.3], [0.7]])
    cpd_fehlzeiten = TabularCPD(variable='Fehlzeiten', variable_card=2, values=[[0.85], [0.15]])

    cpd_pruefungsangst = TabularCPD(
        variable='Pruefungsangst', variable_card=2,
        values=[
            [0.85, 0.40],  # Nein (0)
            [0.15, 0.60]   # Ja (1)
        ],
        evidence=['Fehlzeiten'], evidence_card=[2]
    )

    cpd_verstaendnis = TabularCPD(
        variable='Verstaendnis', variable_card=2,
        values=[
            [0.95, 0.99, 0.70, 0.85, 0.60, 0.80, 0.30, 0.50, 0.50, 0.75, 0.25, 0.45, 0.15, 0.35, 0.05, 0.15], # Gering (0)
            [0.05, 0.01, 0.30, 0.15, 0.40, 0.20, 0.70, 0.50, 0.50, 0.25, 0.75, 0.55, 0.85, 0.65, 0.95, 0.85]  # Hoch (1)
        ],
        evidence=['Vorkenntnisse', 'Hausaufgaben', 'Mitarbeit', 'Fehlzeiten'],
        evidence_card=[2, 2, 2, 2]
    )

    cpd_pruefung = TabularCPD(
        variable='Pruefung_bestanden', variable_card=2,
        values=[
            [0.85, 0.95, 0.05, 0.25],  # Nein (0)
            [0.15, 0.05, 0.95, 0.75]   # Ja (1)
        ],
        evidence=['Verstaendnis', 'Pruefungsangst'], evidence_card=[2, 2]
    )

    model.add_cpds(cpd_vorkenntnisse, cpd_hausaufgaben, cpd_mitarbeit, cpd_fehlzeiten, cpd_pruefungsangst, cpd_verstaendnis, cpd_pruefung)
    return VariableElimination(model)

inference = init_custom_network()

# -----------------------------------------------------------------------------
# 2. OBERFLÄCHEN-LAYOUT CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Schul-Frühwarnsystem", layout="wide")

st.title("🎓 Pädagogisches Prognosemodell (Bayessches Netzwerk)")
st.markdown("Simuliere den Einfluss von Fehlzeiten auf Prüfungsangst, Verständnis und den finalen Prüfungserfolg.")

# Seitenleiste für Evidenz-Eingaben
st.sidebar.header("📋 Schülerdaten eingeben")
st.sidebar.markdown("*Unbekannte Faktoren werden statistisch geschätzt.*")

evidence = {}
display_states = {}

def create_sidebar_input(label, key, options):
    status = st.sidebar.radio(f"{label}:", ["Unbekannt"] + list(options.keys()), key=key)
    if status != "Unbekannt":
        evidence[key] = options[status]
        display_states[key] = status
    else:
        display_states[key] = "Unbekannt"

# Inputs generieren
create_sidebar_input("Vorkenntnisse", "Vorkenntnisse", {"Schlecht": 0, "Gut": 1})
st.sidebar.markdown("---")
create_sidebar_input("Hausaufgaben", "Hausaufgaben", {"Schlecht/Unvollständig": 0, "Gut/Vollständig": 1})
st.sidebar.markdown("---")
create_sidebar_input("Mitarbeit", "Mitarbeit", {"Schlecht": 0, "Gut": 1})
st.sidebar.markdown("---")
create_sidebar_input("Fehlzeiten", "Fehlzeiten", {"Wenig/Normal": 0, "Viele Fehlzeiten": 1})

# -----------------------------------------------------------------------------
# 3. LIVE-INFERENZ BERECHNEN
# -----------------------------------------------------------------------------
# Hilfsfunktion zur Ermittlung der Wahrscheinlichkeiten für alle Knoten
def get_prob(variable):
    if variable in evidence:
        return 100.0 if evidence[variable] == 1 else 0.0
    res = inference.query(variables=[variable], evidence=evidence)
    return res.values[1] * 100

prob_pass = get_prob('Pruefung_bestanden')
prob_verst = get_prob('Verstaendnis')
prob_angst = get_prob('Pruefungsangst')

# -----------------------------------------------------------------------------
# 4. DASHBOARD STRUKTUR (ZWEI SPALTEN)
# -----------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔮 Analyseergebnis")
    
    # Haupt-Metrik: Bestehens-Wahrscheinlichkeit
    if prob_pass >= 75:
        st.success(f"### Wahrscheinlichkeit Bestehen: {prob_pass:.1f}%")
    elif prob_pass >= 40:
        st.warning(f"### Wahrscheinlichkeit Bestehen: {prob_pass:.1f}%")
    else:
        st.error(f"### 🚨 Wahrscheinlichkeit Bestehen: {prob_pass:.1f}%")
    st.progress(int(prob_pass))
    
    st.markdown("---")
    
    # Zwischenknoten visualisieren
    st.markdown(f"**🧠 Hohes Fachverständnis:** `{prob_verst:.1f}%`")
    st.progress(int(prob_verst))
    
    st.markdown(f"**⚡ Akute Prüfungsangst:** `{prob_angst:.1f}%`")
    st.progress(int(prob_angst))

with col2:
    st.subheader("📐 Kausale Verknüpfungen & Netzzustand")
    
    # Graphviz-Netzwerk dynamisch zeichnen
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR', size='7,5')
    
    # Farben basierend auf Zustand vergeben
    def get_node_color(key, default_color="white"):
        return "#A0C4FF" if key in evidence else default_color

    # Knoten hinzufügen
    dot.node('V', f"Vorkenntnisse\n({display_states['Vorkenntnisse']})", style='filled', fillcolor=get_node_color('Vorkenntnisse'))
    dot.node('H', f"Hausaufgaben\n({display_states['Hausaufgaben']})", style='filled', fillcolor=get_node_color('Hausaufgaben'))
    dot.node('M', f"Mitarbeit\n({display_states['Mitarbeit']})", style='filled', fillcolor=get_node_color('Mitarbeit'))
    dot.node('F', f"Fehlzeiten\n({display_states['Fehlzeiten']})", style='filled', fillcolor=get_node_color('Fehlzeiten'))
    
    # Intermediäre Knoten zeigen berechnete Wahrscheinlichkeiten
    dot.node('Verst', f"Verständnis\n(Hoch: {prob_verst:.0f}%)", style='filled', fillcolor="#E2F0CB")
    dot.node('Angst', f"Prüfungsangst\n(Ja: {prob_angst:.0f}%)", style='filled', fillcolor="#FFADAD" if prob_angst > 40 else "#FFFFFC")
    
    # Zielknoten einfärben
    target_color = "#CAFFBF" if prob_pass >= 75 else ("#FDFFB6" if prob_pass >= 40 else "#FFADAD")
    dot.node('Ziel', f"Prüfung bestanden\n(Ja: {prob_pass:.1f}%)", shape='box', style='filled', fillcolor=target_color)
    
    # Verbindungen (Kanten) setzen
    dot.edge('V', 'Verst')
    dot.edge('H', 'Verst')
    dot.edge('M', 'Verst')
    dot.edge('F', 'Verst')
    dot.edge('F', 'Angst')
    dot.edge('Verst', 'Ziel')
    dot.edge('Angst', 'Ziel')
    
    st.graphviz_chart(dot)

# -----------------------------------------------------------------------------
# 5. AKTIONEN & HINWEISE
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📋 Vordefinierte Test-Szenarien")

# Buttons für die schnellen Szenariowechsel
if st.button("🚀 Starte Domino-Effekt Szenario (Viele Fehlzeiten trotz guten Vorkenntnissen)"):
    st.info("Bitte stelle dazu in der linken Leiste ein:\n"
            "- Vorkenntnisse: Gut\n- Hausaufgaben: Schlecht/Unvollständig\n- Mitarbeit: Gut\n- Fehlzeiten: Viele Fehlzeiten")

st.markdown("""
**Analyse des Netzwerks:**
* **Der Fehlzeiten-Effekt:** Wenn du *Fehlzeiten* auf 'Viele' setzt, erhöht sich automatisch die *Prüfungsangst* auf **60%** (Szenariowert deiner CPD) und zeitgleich sinkt die Chance auf ein hohes *Verständnis*. 
* **Erklärbarkeit:** Über die grafischen Prozentwerte siehst du sofort, ob ein schlechtes Abschneiden eher am mangelnden *Verständnis* (z.B. durch nicht gemachte Hausaufgaben) oder an der induzierten *Prüfungsangst* liegt.
""")
