def load_css():

    return """
<style>

/* =========================
BACKGROUND
========================= */

.stApp{
    background-color:#0E1117;
    color:white;
}

/* =========================
SIDEBAR
========================= */

section[data-testid="stSidebar"]{
    background:#161B22;
}

/* =========================
TITLE
========================= */

.main-title{
    font-size:42px;
    color:#00E5FF;
    font-weight:bold;
}

.sub-title{
    font-size:18px;
    color:#BBBBBB;
}

/* =========================
CARDS
========================= */

.metric-card{

    background:#1C2128;

    padding:20px;

    border-radius:15px;

    box-shadow:0px 4px 12px rgba(0,0,0,.4);

    text-align:center;

    border:1px solid #30363D;

    transition:.3s;
}

.metric-card:hover{

    transform:scale(1.03);

    border:1px solid cyan;

}

/* =========================
SECTION TITLE
========================= */

.section-title{

    color:#00E5FF;

    font-size:28px;

    font-weight:bold;

}

/* =========================
FOOTER
========================= */

.footer{

text-align:center;

color:gray;

padding-top:50px;

}

</style>
"""