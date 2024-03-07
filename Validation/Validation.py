import hist
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
from pathlib import Path
import pickle

from .fit import fitCruijff, cruijff


def createHists():
    max_E, max_E_tot = 100, 800
    max_ratio = 1.5
    bins = 200

    return dict(
        h_pred = hist.Hist(hist.axis.Regular(bins, 0., max_E, name="pred_energy", label="Predicted trackster energy (GeV)")),
        h_reco = hist.Hist(hist.axis.Regular(bins, 0., max_E, name="reco_energy", label="Trackster raw energy (GeV)")),
        h_reco_tot = hist.Hist(hist.axis.Regular(bins, 0., max_E_tot, name="reco_energy_tot", label="Total trackster raw energy (GeV)")),
        h_pred_tot = hist.Hist(hist.axis.Regular(bins, 0., max_E_tot, name="pred_energy_tot", label="Predicted energy for full endcap (GeV)")),
        h_cp = hist.Hist(hist.axis.Regular(bins, 0., max_E_tot, name="cp_energy", label="CaloParticle (true) energy (GeV)")),
        h_reco_tot_over_cp = hist.Hist(hist.axis.Regular(bins, 0., max_ratio, name="reco_tot_over_cp", label="Total trackster raw energy / CaloParticle energy")),
        h_pred_tot_over_cp = hist.Hist(hist.axis.Regular(bins, 0., max_ratio, name="pred_tot_over_cp", label="Total trackster predicted energy / CaloParticle energy"))
    )


def write_histograms(X,y,prediction):
    hists = createHists()
    hists["h_reco_tot"].fill(X["tkx_energy"])
    hists["h_pred_tot"].fill(prediction)
    hists["h_cp"].fill(y)
    hists["h_reco_tot_over_cp"].fill(X["tkx_energy"]/y)
    hists["h_pred_tot_over_cp"].fill(prediction / y)
    return hists
    

def plotTracksterEnergies(hists):
    plt.figure(figsize=(9, 9))
    hep.histplot([hists["h_reco"], hists["h_pred"]], label=["Raw trackster energy", "Predicted trackster energy"])
    plt.ylabel("Tracksters")
    plt.xlabel("Trackster energy (GeV)")
    plt.xlim(0, 50)
    plt.legend(loc="upper right")

def plotFullEnergies(hists):
    plt.figure()
    hep.histplot([hists["h_reco_tot"], hists["h_pred_tot"], hists["h_cp"]], yerr=False, label=["Sum of raw trackster energy", "Sum of predicted energies", "CaloParticle (true) energy"])
    plt.ylabel("Events")
    plt.xlabel("Energy in endcap (GeV)")
    plt.legend()

def plotRatioOverCP(hists):
    plt.figure()
    hep.histplot([hists["h_reco_tot_over_cp"], hists["h_pred_tot_over_cp"]], yerr=False, label=["Sum of raw trackster energy", "Sum of predicted trackster energy"])
    
    def plotFit(h:hist.Hist):
        fitRes = fitCruijff(h)
        params = fitRes.params
        x_plotFct = np.linspace(h.axes[0].centers[0], h.axes[0].centers[-1],500)
        plt.plot(x_plotFct,cruijff(x_plotFct,*params.makeTuple()), 
            label=f"Cruijff fit\n$\sigma={(params.sigmaL+params.sigmaR)/2:.3f}$, $\mu={params.m:.3f}$, " +r"$\frac{\sigma}{\mu}=" + f"{(params.sigmaL+params.sigmaR)/(2*params.m):.3f}$")

    plotFit(hists["h_reco_tot_over_cp"])
    plotFit(hists["h_pred_tot_over_cp"])

    plt.ylabel("Events")
    plt.xlabel("Ratio over CaloParticle energy")
    plt.legend()


plotsToSave = [plotFullEnergies, plotRatioOverCP]

def doFullValidation(X, y, prediction,model_path):
    hists = write_histograms(X, y, prediction)
    with open(Path(model_path).with_suffix(".hists.pkl"), "wb") as f:
        pickle.dump(hists, f)
    
    for plotFct in plotsToSave:
        plotFct(hists)
        plt.savefig(Path(model_path).with_suffix("." + plotFct.__name__ + ".png"))