import pandas as pd
from src.common import vintage_from_numeric
from src.scenario.engine import apply_manpower_scenario


def test_vintage_vlookup_equivalent():
    assert vintage_from_numeric(0)=="0-3 M"
    assert vintage_from_numeric(3.99)=="0-3 M"
    assert vintage_from_numeric(4)=="4-6 M"
    assert vintage_from_numeric(7)=="7-12 M"
    assert vintage_from_numeric(13)=="13-24 M"
    assert vintage_from_numeric(24)=="13-24 M"
    assert vintage_from_numeric(24.1)=="24+ M"


def test_manpower_scenario_propagates_opening():
    b=pd.DataFrame({"Channel":["A","A"],"Zone":["East","East"],"Month":pd.to_datetime(["2026-04-01","2026-05-01"]),"Opening":[100,105],"Closing":[105,108],"Exit":[5,7],"ForecastHiring":[10,10]})
    out=apply_manpower_scenario(b,[{"Channel":"A","Zone":"East","Month":"2026-04-01","field":"Hiring","value":20}])
    assert out.iloc[0].ScenarioClosing==115
    assert out.iloc[1].ScenarioOpening==115
