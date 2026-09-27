from __future__ import annotations

import json, os
import pandas as pd


def deterministic_summary(manpower=None, scenario=None, nop=None, ats=None):
    facts={}
    if manpower is not None and not manpower.empty:
        ch=manpower.groupby("Channel",as_index=False).agg(Closing=("ForecastClosing","sum"),Exit=("ForecastExit","sum"))
        facts["top_closing_channels"]=ch.nlargest(5,"Closing").to_dict("records")
        facts["top_exit_channels"]=ch.nlargest(5,"Exit").to_dict("records")
    if scenario is not None and not scenario.empty and "ClosingDelta" in scenario:
        facts["largest_scenario_closing_impacts"]=scenario.reindex(scenario.ClosingDelta.abs().sort_values(ascending=False).index).head(10).to_dict("records")
    if nop is not None and not nop.empty:
        facts["top_nop_channels"]=nop.groupby("Channel",as_index=False)["ForecastNOP"].sum().nlargest(5,"ForecastNOP").to_dict("records")
    if ats is not None and not ats.empty:
        facts["top_ats_channels"]=ats.groupby("Channel",as_index=False)["ForecastATS"].mean().nlargest(5,"ForecastATS").to_dict("records")
    return facts


def generate_llm_insights(facts: dict):
    key=os.getenv("AZURE_OPENAI_API_KEY"); base=os.getenv("AZURE_OPENAI_BASE_URL"); model=os.getenv("AZURE_OPENAI_MODEL","gpt-4o")
    if not key or not base:
        return {"mode":"deterministic-only","facts":facts,"message":"Azure OpenAI environment variables are not configured."}
    from openai import OpenAI
    client=OpenAI(api_key=key,base_url=base)
    prompt=("You are an insurance business-planning analytics assistant. Explain only the supplied computed facts. "
            "Do not invent numbers and do not replace the ML forecast. Highlight where manpower, exit/attrition, NOP or ATS appear weak, "
            "what changed in the scenario, and practical questions business should investigate. Return concise JSON with summary, risks, opportunities, actions.\nFACTS:\n"+json.dumps(facts,default=str))
    response=client.responses.create(model=model,input=prompt)
    return {"mode":"llm","text":response.output_text,"facts":facts}
