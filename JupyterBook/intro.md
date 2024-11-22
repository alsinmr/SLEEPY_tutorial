# Welcome to the pyDIFRATE tutorial

pyDIFRATE is software originally developed for analyzing NMR relaxation rate constants with minimal assumptions about the form of the correlation function. It uses the  "Detectors" approach, to return amplitudes of motion for timescale-specific windows. It has been broadly expanded to also analyze MD simulations. These analyses include direct comparison with NMR results, timescale-specific cross-correlation of motion, and separation of motion via frames (ROMANCE analysis). A principal component analysis module is also underway.

These features are coupled with powerful visualization methods, including 3D visualization on the structure with [ChimeraX](https://www.cgl.ucsf.edu/chimerax/), and with a file management system to keep track of data processing steps and help with data sorting and comparison.

The tutorial here provides explanation of the detectors motivation, and some basic explanation of the theory, as well as example analyses. A series of templates are also provided into which one may insert ones own data, and can either be downloaded or run on [Google Colab](https://colab.research.google.com/?utm_source=scs-index). Running on Colab means you do not need to provide your data to our servers, nor is it stored by Google beyond the lifetime of the session.

```{tableofcontents}
```

**HET-s(218-289) Dynamics**
*NMR data analyzed via detector analysis and plotted with [NGLview](http://nglviewer.org/) (built into pyDR). The detector near 6 ns is shown*

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/jqueryui/1.12.0/jquery-ui.css">


<!-- Load require.js. Delete this if your page already loads require.js -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js" integrity="sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA=" crossorigin="anonymous"></script>
<script src="https://unpkg.com/@jupyter-widgets/html-manager@^0.20.0/dist/embed-amd.js" crossorigin="anonymous"></script>

<script type="application/vnd.jupyter.widget-state+json">
{
  "version_major": 2,
  "version_minor": 0,
  "state": {
    "7931c71a4b2449faae9f1a62d61642e9": {
      "model_name": "LayoutModel",
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "state": {}
    },
    "5f6216975e6646e2b04accd91c70446a": {
      "model_name": "ColormakerRegistryModel",
      "model_module": "nglview-js-widgets",
      "model_module_version": "3.0.1",
      "state": {
        "_dom_classes": [],
        "_msg_ar": [
          {
            "type": "callMethod",
            "methodName": "executeCode",
            "args": [
              "\n            var schemeId = NGL.ColormakerRegistry.addScheme(function (params) {\n                this.atomColor = function (atom) {\n  if  (atom.serial == 22){\n  return 0xD0B38B\n  } else if  (atom.serial == 23){\n  return 0xD0B38B\n  } else if  (atom.serial == 24){\n  return 0xD0B38B\n  } else if  (atom.serial == 25){\n  return 0xD0B38B\n  } else if  (atom.serial == 26){\n  return 0xD0B38B\n  } else if  (atom.serial == 27){\n  return 0xD0B38B\n  } else if  (atom.serial == 32){\n  return 0xB2AF79\n  } else if  (atom.serial == 33){\n  return 0xB2AF79\n  } else if  (atom.serial == 34){\n  return 0xB2AF79\n  } else if  (atom.serial == 35){\n  return 0xB2AF79\n  } else if  (atom.serial == 36){\n  return 0xB2AF79\n  } else if  (atom.serial == 37){\n  return 0xAFAF78\n  } else if  (atom.serial == 38){\n  return 0xADAF76\n  } else if  (atom.serial == 39){\n  return 0xADAF76\n  } else if  (atom.serial == 40){\n  return 0xADAF76\n  } else if  (atom.serial == 41){\n  return 0xADAF76\n  } else if  (atom.serial == 42){\n  return 0xB5B07B\n  } else if  (atom.serial == 43){\n  return 0xBCB07F\n  } else if  (atom.serial == 44){\n  return 0xBCB07F\n  } else if  (atom.serial == 45){\n  return 0xBCB07F\n  } else if  (atom.serial == 46){\n  return 0xBCB07F\n  } else if  (atom.serial == 47){\n  return 0xA3AD71\n  } else if  (atom.serial == 48){\n  return 0x8AAB62\n  } else if  (atom.serial == 49){\n  return 0x8AAB62\n  } else if  (atom.serial == 50){\n  return 0x8AAB62\n  } else if  (atom.serial == 51){\n  return 0x8AAB62\n  } else if  (atom.serial == 52){\n  return 0x94AC68\n  } else if  (atom.serial == 53){\n  return 0x9EAD6E\n  } else if  (atom.serial == 54){\n  return 0x9EAD6E\n  } else if  (atom.serial == 55){\n  return 0x9EAD6E\n  } else if  (atom.serial == 56){\n  return 0x9EAD6E\n  } else if  (atom.serial == 57){\n  return 0xB8B07C\n  } else if  (atom.serial == 58){\n  return 0xD1B38B\n  } else if  (atom.serial == 59){\n  return 0xD1B38B\n  } else if  (atom.serial == 60){\n  return 0xD1B38B\n  } else if  (atom.serial == 61){\n  return 0xD1B38B\n  } else if  (atom.serial == 62){\n  return 0xC6B185\n  } else if  (atom.serial == 63){\n  return 0xBAB07E\n  } else if  (atom.serial == 64){\n  return 0xBAB07E\n  } else if  (atom.serial == 65){\n  return 0xBAB07E\n  } else if  (atom.serial == 66){\n  return 0xBAB07E\n  } else if  (atom.serial == 67){\n  return 0xB5B07B\n  } else if  (atom.serial == 68){\n  return 0xAFAF78\n  } else if  (atom.serial == 69){\n  return 0xAFAF78\n  } else if  (atom.serial == 70){\n  return 0xAFAF78\n  } else if  (atom.serial == 71){\n  return 0xAFAF78\n  } else if  (atom.serial == 72){\n  return 0xA0AD6F\n  } else if  (atom.serial == 73){\n  return 0x90AB66\n  } else if  (atom.serial == 74){\n  return 0x90AB66\n  } else if  (atom.serial == 75){\n  return 0x90AB66\n  } else if  (atom.serial == 76){\n  return 0x90AB66\n  } else if  (atom.serial == 77){\n  return 0x8EAB65\n  } else if  (atom.serial == 78){\n  return 0x8CAB63\n  } else if  (atom.serial == 79){\n  return 0x8CAB63\n  } else if  (atom.serial == 80){\n  return 0x8CAB63\n  } else if  (atom.serial == 81){\n  return 0x8CAB63\n  } else if  (atom.serial == 82){\n  return 0x94AC68\n  } else if  (atom.serial == 83){\n  return 0x9BAD6C\n  } else if  (atom.serial == 84){\n  return 0x9BAD6C\n  } else if  (atom.serial == 85){\n  return 0x9BAD6C\n  } else if  (atom.serial == 86){\n  return 0x9BAD6C\n  } else if  (atom.serial == 87){\n  return 0x9AAC6C\n  } else if  (atom.serial == 88){\n  return 0x9AAC6B\n  } else if  (atom.serial == 89){\n  return 0x9AAC6B\n  } else if  (atom.serial == 90){\n  return 0x9AAC6B\n  } else if  (atom.serial == 91){\n  return 0x9AAC6B\n  } else if  (atom.serial == 92){\n  return 0x9EAD6E\n  } else if  (atom.serial == 93){\n  return 0xA2AD70\n  } else if  (atom.serial == 94){\n  return 0xA2AD70\n  } else if  (atom.serial == 95){\n  return 0xA2AD70\n  } else if  (atom.serial == 96){\n  return 0xA2AD70\n  } else if  (atom.serial == 97){\n  return 0xAAAE75\n  } else if  (atom.serial == 98){\n  return 0xB2AF79\n  } else if  (atom.serial == 99){\n  return 0xB2AF79\n  } else if  (atom.serial == 100){\n  return 0xB2AF79\n  } else if  (atom.serial == 101){\n  return 0xB2AF79\n  } else if  (atom.serial == 102){\n  return 0xB8B07D\n  } else if  (atom.serial == 103){\n  return 0xBEB180\n  } else if  (atom.serial == 104){\n  return 0xBEB180\n  } else if  (atom.serial == 105){\n  return 0xBEB180\n  } else if  (atom.serial == 106){\n  return 0xBEB180\n  } else if  (atom.serial == 107){\n  return 0xC4B184\n  } else if  (atom.serial == 108){\n  return 0xCAB287\n  } else if  (atom.serial == 109){\n  return 0xCAB287\n  } else if  (atom.serial == 110){\n  return 0xCAB287\n  } else if  (atom.serial == 111){\n  return 0xCAB287\n  } else if  (atom.serial == 112){\n  return 0xC4B183\n  } else if  (atom.serial == 113){\n  return 0xBDB180\n  } else if  (atom.serial == 114){\n  return 0xBDB180\n  } else if  (atom.serial == 115){\n  return 0xBDB180\n  } else if  (atom.serial == 116){\n  return 0xBDB180\n  } else if  (atom.serial == 117){\n  return 0xB9B07D\n  } else if  (atom.serial == 118){\n  return 0xB5B07B\n  } else if  (atom.serial == 119){\n  return 0xB5B07B\n  } else if  (atom.serial == 120){\n  return 0xB5B07B\n  } else if  (atom.serial == 121){\n  return 0xB5B07B\n  } else if  (atom.serial == 122){\n  return 0xC0B181\n  } else if  (atom.serial == 123){\n  return 0xCBB288\n  } else if  (atom.serial == 124){\n  return 0xCBB288\n  } else if  (atom.serial == 125){\n  return 0xCBB288\n  } else if  (atom.serial == 126){\n  return 0xCBB288\n  } else if  (atom.serial == 127){\n  return 0xC6B285\n  } else if  (atom.serial == 128){\n  return 0xC1B182\n  } else if  (atom.serial == 129){\n  return 0xC1B182\n  } else if  (atom.serial == 130){\n  return 0xC1B182\n  } else if  (atom.serial == 131){\n  return 0xC1B182\n  } else if  (atom.serial == 132){\n  return 0xAAAE75\n  } else if  (atom.serial == 133){\n  return 0x94AC68\n  } else if  (atom.serial == 134){\n  return 0x94AC68\n  } else if  (atom.serial == 135){\n  return 0x94AC68\n  } else if  (atom.serial == 136){\n  return 0x94AC68\n  } else if  (atom.serial == 137){\n  return 0x93AC68\n  } else if  (atom.serial == 138){\n  return 0x93AC68\n  } else if  (atom.serial == 139){\n  return 0x93AC68\n  } else if  (atom.serial == 140){\n  return 0x93AC68\n  } else if  (atom.serial == 141){\n  return 0x93AC68\n  } else if  (atom.serial == 142){\n  return 0xAFAF78\n  } else if  (atom.serial == 143){\n  return 0xCCB288\n  } else if  (atom.serial == 144){\n  return 0xCCB288\n  } else if  (atom.serial == 145){\n  return 0xCCB288\n  } else if  (atom.serial == 146){\n  return 0xCCB288\n  } else if  (atom.serial == 147){\n  return 0x7CA95A\n  } else if  (atom.serial == 148){\n  return 0x2CA02C\n  } else if  (atom.serial == 149){\n  return 0x2CA02C\n  } else if  (atom.serial == 150){\n  return 0x2CA02C\n  } else if  (atom.serial == 151){\n  return 0x2CA02C\n  } else if  (atom.serial == 152){\n  return 0x2CA02C\n  } else if  (atom.serial == 207){\n  return 0x66A64D\n  } else if  (atom.serial == 208){\n  return 0x66A64D\n  } else if  (atom.serial == 209){\n  return 0x66A64D\n  } else if  (atom.serial == 210){\n  return 0x66A64D\n  } else if  (atom.serial == 211){\n  return 0x66A64D\n  } else if  (atom.serial == 212){\n  return 0x83AA5E\n  } else if  (atom.serial == 213){\n  return 0xA0AD6F\n  } else if  (atom.serial == 214){\n  return 0xA0AD6F\n  } else if  (atom.serial == 215){\n  return 0xA0AD6F\n  } else if  (atom.serial == 216){\n  return 0xA0AD6F\n  } else if  (atom.serial == 217){\n  return 0xB8B07D\n  } else if  (atom.serial == 218){\n  return 0xCFB38A\n  } else if  (atom.serial == 219){\n  return 0xCFB38A\n  } else if  (atom.serial == 220){\n  return 0xCFB38A\n  } else if  (atom.serial == 221){\n  return 0xCFB38A\n  } else if  (atom.serial == 222){\n  return 0xC7B285\n  } else if  (atom.serial == 223){\n  return 0xBEB180\n  } else if  (atom.serial == 224){\n  return 0xBEB180\n  } else if  (atom.serial == 225){\n  return 0xBEB180\n  } else if  (atom.serial == 226){\n  return 0xBEB180\n  } else if  (atom.serial == 227){\n  return 0xB4AF7B\n  } else if  (atom.serial == 228){\n  return 0xABAE75\n  } else if  (atom.serial == 229){\n  return 0xABAE75\n  } else if  (atom.serial == 230){\n  return 0xABAE75\n  } else if  (atom.serial == 231){\n  return 0xABAE75\n  } else if  (atom.serial == 232){\n  return 0xB4AF7A\n  } else if  (atom.serial == 233){\n  return 0xBDB180\n  } else if  (atom.serial == 234){\n  return 0xBDB180\n  } else if  (atom.serial == 235){\n  return 0xBDB180\n  } else if  (atom.serial == 236){\n  return 0xBDB180\n  } else if  (atom.serial == 237){\n  return 0xB9B07D\n  } else if  (atom.serial == 238){\n  return 0xB5B07B\n  } else if  (atom.serial == 239){\n  return 0xB5B07B\n  } else if  (atom.serial == 240){\n  return 0xB5B07B\n  } else if  (atom.serial == 241){\n  return 0xB5B07B\n  } else if  (atom.serial == 242){\n  return 0xB1AF79\n  } else if  (atom.serial == 243){\n  return 0xAEAF77\n  } else if  (atom.serial == 244){\n  return 0xAEAF77\n  } else if  (atom.serial == 245){\n  return 0xAEAF77\n  } else if  (atom.serial == 246){\n  return 0xAEAF77\n  } else if  (atom.serial == 247){\n  return 0xB4AF7A\n  } else if  (atom.serial == 248){\n  return 0xB9B07D\n  } else if  (atom.serial == 249){\n  return 0xB9B07D\n  } else if  (atom.serial == 250){\n  return 0xB9B07D\n  } else if  (atom.serial == 251){\n  return 0xB9B07D\n  } else if  (atom.serial == 252){\n  return 0xB6B07C\n  } else if  (atom.serial == 253){\n  return 0xB3AF7A\n  } else if  (atom.serial == 254){\n  return 0xB3AF7A\n  } else if  (atom.serial == 255){\n  return 0xB3AF7A\n  } else if  (atom.serial == 256){\n  return 0xB3AF7A\n  } else if  (atom.serial == 257){\n  return 0xA6AE72\n  } else if  (atom.serial == 258){\n  return 0x99AC6B\n  } else if  (atom.serial == 259){\n  return 0x99AC6B\n  } else if  (atom.serial == 260){\n  return 0x99AC6B\n  } else if  (atom.serial == 261){\n  return 0x99AC6B\n  } else if  (atom.serial == 262){\n  return 0xB5B07B\n  } else if  (atom.serial == 263){\n  return 0xD1B38B\n  } else if  (atom.serial == 264){\n  return 0xD1B38B\n  } else if  (atom.serial == 265){\n  return 0xD1B38B\n  } else if  (atom.serial == 266){\n  return 0xD1B38B\n  } else if  (atom.serial == 267){\n  return 0xABAE75\n  } else if  (atom.serial == 268){\n  return 0x86AA60\n  } else if  (atom.serial == 269){\n  return 0x86AA60\n  } else if  (atom.serial == 270){\n  return 0x86AA60\n  } else if  (atom.serial == 271){\n  return 0x86AA60\n  } else if  (atom.serial == 272){\n  return 0x88AA61\n  } else if  (atom.serial == 273){\n  return 0x89AA62\n  } else if  (atom.serial == 274){\n  return 0x89AA62\n  } else if  (atom.serial == 275){\n  return 0x89AA62\n  } else if  (atom.serial == 276){\n  return 0x89AA62\n  } else if  (atom.serial == 277){\n  return 0xA0AD6F\n  } else if  (atom.serial == 278){\n  return 0xB6B07B\n  } else if  (atom.serial == 279){\n  return 0xB6B07B\n  } else if  (atom.serial == 280){\n  return 0xB6B07B\n  } else if  (atom.serial == 281){\n  return 0xB6B07B\n  } else if  (atom.serial == 282){\n  return 0xBCB07F\n  } else if  (atom.serial == 283){\n  return 0xC1B182\n  } else if  (atom.serial == 284){\n  return 0xC1B182\n  } else if  (atom.serial == 285){\n  return 0xC1B182\n  } else if  (atom.serial == 286){\n  return 0xC1B182\n  } else if  (atom.serial == 287){\n  return 0xC6B285\n  } else if  (atom.serial == 288){\n  return 0xCAB287\n  } else if  (atom.serial == 289){\n  return 0xCAB287\n  } else if  (atom.serial == 290){\n  return 0xCAB287\n  } else if  (atom.serial == 291){\n  return 0xCAB287\n  } else if  (atom.serial == 292){\n  return 0xC3B183\n  } else if  (atom.serial == 293){\n  return 0xBDB07F\n  } else if  (atom.serial == 294){\n  return 0xBDB07F\n  } else if  (atom.serial == 295){\n  return 0xBDB07F\n  } else if  (atom.serial == 296){\n  return 0xBDB07F\n  } else if  (atom.serial == 297){\n  return 0xC4B184\n  } else if  (atom.serial == 298){\n  return 0xCCB288\n  } else if  (atom.serial == 299){\n  return 0xCCB288\n  } else if  (atom.serial == 300){\n  return 0xCCB288\n  } else if  (atom.serial == 301){\n  return 0xCCB288\n  } else if  (atom.serial == 302){\n  return 0xB3AF7A\n  } else if  (atom.serial == 303){\n  return 0x9BAC6C\n  } else if  (atom.serial == 304){\n  return 0x9BAC6C\n  } else if  (atom.serial == 305){\n  return 0x9BAC6C\n  } else if  (atom.serial == 306){\n  return 0x9BAC6C\n  } else if  (atom.serial == 307){\n  return 0xB1AF78\n  } else if  (atom.serial == 308){\n  return 0xC7B285\n  } else if  (atom.serial == 309){\n  return 0xC7B285\n  } else if  (atom.serial == 310){\n  return 0xC7B285\n  } else if  (atom.serial == 311){\n  return 0xC7B285\n  } else if  (atom.serial == 312){\n  return 0xB5B07B\n  } else if  (atom.serial == 313){\n  return 0xA3AD71\n  } else if  (atom.serial == 314){\n  return 0xA3AD71\n  } else if  (atom.serial == 315){\n  return 0xA3AD71\n  } else if  (atom.serial == 316){\n  return 0xA3AD71\n  } else if  (atom.serial == 317){\n  return 0xAEAF77\n  } else if  (atom.serial == 318){\n  return 0xB9B07D\n  } else if  (atom.serial == 319){\n  return 0xB9B07D\n  } else if  (atom.serial == 320){\n  return 0xB9B07D\n  } else if  (atom.serial == 321){\n  return 0xB9B07D\n  } else if  (atom.serial == 322){\n  return 0xB9B07D\n  } else if  (atom.serial == 327){\n  return 0x8AAB62\n  } else if  (atom.serial == 328){\n  return 0x8AAB62\n  } else if  (atom.serial == 329){\n  return 0x8AAB62\n  } else if  (atom.serial == 330){\n  return 0x8AAB62\n  } else if  (atom.serial == 331){\n  return 0x8AAB62\n  } else if  (atom.serial == 332){\n  return 0x8AAB62\n  } else  {\n  return 0xD1B38B\n  }\n }\n \n            })\n            \n            this._updateId(schemeId, 'color1')\n        "
            ],
            "kwargs": {}
          },
          {
            "type": "callMethod",
            "methodName": "executeCode",
            "args": [
              "\n            var schemeId = NGL.ColormakerRegistry.addScheme(function (params) {\n                this.atomColor = function (atom) {\n  if  (atom.serial == 22){\n  return 0xD0B38B\n  } else if  (atom.serial == 23){\n  return 0xD0B38B\n  } else if  (atom.serial == 24){\n  return 0xD0B38B\n  } else if  (atom.serial == 25){\n  return 0xD0B38B\n  } else if  (atom.serial == 26){\n  return 0xD0B38B\n  } else if  (atom.serial == 27){\n  return 0xD0B38B\n  } else if  (atom.serial == 32){\n  return 0xB2AF79\n  } else if  (atom.serial == 33){\n  return 0xB2AF79\n  } else if  (atom.serial == 34){\n  return 0xB2AF79\n  } else if  (atom.serial == 35){\n  return 0xB2AF79\n  } else if  (atom.serial == 36){\n  return 0xB2AF79\n  } else if  (atom.serial == 37){\n  return 0xAFAF78\n  } else if  (atom.serial == 38){\n  return 0xADAF76\n  } else if  (atom.serial == 39){\n  return 0xADAF76\n  } else if  (atom.serial == 40){\n  return 0xADAF76\n  } else if  (atom.serial == 41){\n  return 0xADAF76\n  } else if  (atom.serial == 42){\n  return 0xB5B07B\n  } else if  (atom.serial == 43){\n  return 0xBCB07F\n  } else if  (atom.serial == 44){\n  return 0xBCB07F\n  } else if  (atom.serial == 45){\n  return 0xBCB07F\n  } else if  (atom.serial == 46){\n  return 0xBCB07F\n  } else if  (atom.serial == 47){\n  return 0xA3AD71\n  } else if  (atom.serial == 48){\n  return 0x8AAB62\n  } else if  (atom.serial == 49){\n  return 0x8AAB62\n  } else if  (atom.serial == 50){\n  return 0x8AAB62\n  } else if  (atom.serial == 51){\n  return 0x8AAB62\n  } else if  (atom.serial == 52){\n  return 0x94AC68\n  } else if  (atom.serial == 53){\n  return 0x9EAD6E\n  } else if  (atom.serial == 54){\n  return 0x9EAD6E\n  } else if  (atom.serial == 55){\n  return 0x9EAD6E\n  } else if  (atom.serial == 56){\n  return 0x9EAD6E\n  } else if  (atom.serial == 57){\n  return 0xB8B07C\n  } else if  (atom.serial == 58){\n  return 0xD1B38B\n  } else if  (atom.serial == 59){\n  return 0xD1B38B\n  } else if  (atom.serial == 60){\n  return 0xD1B38B\n  } else if  (atom.serial == 61){\n  return 0xD1B38B\n  } else if  (atom.serial == 62){\n  return 0xC6B185\n  } else if  (atom.serial == 63){\n  return 0xBAB07E\n  } else if  (atom.serial == 64){\n  return 0xBAB07E\n  } else if  (atom.serial == 65){\n  return 0xBAB07E\n  } else if  (atom.serial == 66){\n  return 0xBAB07E\n  } else if  (atom.serial == 67){\n  return 0xB5B07B\n  } else if  (atom.serial == 68){\n  return 0xAFAF78\n  } else if  (atom.serial == 69){\n  return 0xAFAF78\n  } else if  (atom.serial == 70){\n  return 0xAFAF78\n  } else if  (atom.serial == 71){\n  return 0xAFAF78\n  } else if  (atom.serial == 72){\n  return 0xA0AD6F\n  } else if  (atom.serial == 73){\n  return 0x90AB66\n  } else if  (atom.serial == 74){\n  return 0x90AB66\n  } else if  (atom.serial == 75){\n  return 0x90AB66\n  } else if  (atom.serial == 76){\n  return 0x90AB66\n  } else if  (atom.serial == 77){\n  return 0x8EAB65\n  } else if  (atom.serial == 78){\n  return 0x8CAB63\n  } else if  (atom.serial == 79){\n  return 0x8CAB63\n  } else if  (atom.serial == 80){\n  return 0x8CAB63\n  } else if  (atom.serial == 81){\n  return 0x8CAB63\n  } else if  (atom.serial == 82){\n  return 0x94AC68\n  } else if  (atom.serial == 83){\n  return 0x9BAD6C\n  } else if  (atom.serial == 84){\n  return 0x9BAD6C\n  } else if  (atom.serial == 85){\n  return 0x9BAD6C\n  } else if  (atom.serial == 86){\n  return 0x9BAD6C\n  } else if  (atom.serial == 87){\n  return 0x9AAC6C\n  } else if  (atom.serial == 88){\n  return 0x9AAC6B\n  } else if  (atom.serial == 89){\n  return 0x9AAC6B\n  } else if  (atom.serial == 90){\n  return 0x9AAC6B\n  } else if  (atom.serial == 91){\n  return 0x9AAC6B\n  } else if  (atom.serial == 92){\n  return 0x9EAD6E\n  } else if  (atom.serial == 93){\n  return 0xA2AD70\n  } else if  (atom.serial == 94){\n  return 0xA2AD70\n  } else if  (atom.serial == 95){\n  return 0xA2AD70\n  } else if  (atom.serial == 96){\n  return 0xA2AD70\n  } else if  (atom.serial == 97){\n  return 0xAAAE75\n  } else if  (atom.serial == 98){\n  return 0xB2AF79\n  } else if  (atom.serial == 99){\n  return 0xB2AF79\n  } else if  (atom.serial == 100){\n  return 0xB2AF79\n  } else if  (atom.serial == 101){\n  return 0xB2AF79\n  } else if  (atom.serial == 102){\n  return 0xB8B07D\n  } else if  (atom.serial == 103){\n  return 0xBEB180\n  } else if  (atom.serial == 104){\n  return 0xBEB180\n  } else if  (atom.serial == 105){\n  return 0xBEB180\n  } else if  (atom.serial == 106){\n  return 0xBEB180\n  } else if  (atom.serial == 107){\n  return 0xC4B184\n  } else if  (atom.serial == 108){\n  return 0xCAB287\n  } else if  (atom.serial == 109){\n  return 0xCAB287\n  } else if  (atom.serial == 110){\n  return 0xCAB287\n  } else if  (atom.serial == 111){\n  return 0xCAB287\n  } else if  (atom.serial == 112){\n  return 0xC4B183\n  } else if  (atom.serial == 113){\n  return 0xBDB180\n  } else if  (atom.serial == 114){\n  return 0xBDB180\n  } else if  (atom.serial == 115){\n  return 0xBDB180\n  } else if  (atom.serial == 116){\n  return 0xBDB180\n  } else if  (atom.serial == 117){\n  return 0xB9B07D\n  } else if  (atom.serial == 118){\n  return 0xB5B07B\n  } else if  (atom.serial == 119){\n  return 0xB5B07B\n  } else if  (atom.serial == 120){\n  return 0xB5B07B\n  } else if  (atom.serial == 121){\n  return 0xB5B07B\n  } else if  (atom.serial == 122){\n  return 0xC0B181\n  } else if  (atom.serial == 123){\n  return 0xCBB288\n  } else if  (atom.serial == 124){\n  return 0xCBB288\n  } else if  (atom.serial == 125){\n  return 0xCBB288\n  } else if  (atom.serial == 126){\n  return 0xCBB288\n  } else if  (atom.serial == 127){\n  return 0xC6B285\n  } else if  (atom.serial == 128){\n  return 0xC1B182\n  } else if  (atom.serial == 129){\n  return 0xC1B182\n  } else if  (atom.serial == 130){\n  return 0xC1B182\n  } else if  (atom.serial == 131){\n  return 0xC1B182\n  } else if  (atom.serial == 132){\n  return 0xAAAE75\n  } else if  (atom.serial == 133){\n  return 0x94AC68\n  } else if  (atom.serial == 134){\n  return 0x94AC68\n  } else if  (atom.serial == 135){\n  return 0x94AC68\n  } else if  (atom.serial == 136){\n  return 0x94AC68\n  } else if  (atom.serial == 137){\n  return 0x93AC68\n  } else if  (atom.serial == 138){\n  return 0x93AC68\n  } else if  (atom.serial == 139){\n  return 0x93AC68\n  } else if  (atom.serial == 140){\n  return 0x93AC68\n  } else if  (atom.serial == 141){\n  return 0x93AC68\n  } else if  (atom.serial == 142){\n  return 0xAFAF78\n  } else if  (atom.serial == 143){\n  return 0xCCB288\n  } else if  (atom.serial == 144){\n  return 0xCCB288\n  } else if  (atom.serial == 145){\n  return 0xCCB288\n  } else if  (atom.serial == 146){\n  return 0xCCB288\n  } else if  (atom.serial == 147){\n  return 0x7CA95A\n  } else if  (atom.serial == 148){\n  return 0x2CA02C\n  } else if  (atom.serial == 149){\n  return 0x2CA02C\n  } else if  (atom.serial == 150){\n  return 0x2CA02C\n  } else if  (atom.serial == 151){\n  return 0x2CA02C\n  } else if  (atom.serial == 152){\n  return 0x2CA02C\n  } else if  (atom.serial == 207){\n  return 0x66A64D\n  } else if  (atom.serial == 208){\n  return 0x66A64D\n  } else if  (atom.serial == 209){\n  return 0x66A64D\n  } else if  (atom.serial == 210){\n  return 0x66A64D\n  } else if  (atom.serial == 211){\n  return 0x66A64D\n  } else if  (atom.serial == 212){\n  return 0x83AA5E\n  } else if  (atom.serial == 213){\n  return 0xA0AD6F\n  } else if  (atom.serial == 214){\n  return 0xA0AD6F\n  } else if  (atom.serial == 215){\n  return 0xA0AD6F\n  } else if  (atom.serial == 216){\n  return 0xA0AD6F\n  } else if  (atom.serial == 217){\n  return 0xB8B07D\n  } else if  (atom.serial == 218){\n  return 0xCFB38A\n  } else if  (atom.serial == 219){\n  return 0xCFB38A\n  } else if  (atom.serial == 220){\n  return 0xCFB38A\n  } else if  (atom.serial == 221){\n  return 0xCFB38A\n  } else if  (atom.serial == 222){\n  return 0xC7B285\n  } else if  (atom.serial == 223){\n  return 0xBEB180\n  } else if  (atom.serial == 224){\n  return 0xBEB180\n  } else if  (atom.serial == 225){\n  return 0xBEB180\n  } else if  (atom.serial == 226){\n  return 0xBEB180\n  } else if  (atom.serial == 227){\n  return 0xB4AF7B\n  } else if  (atom.serial == 228){\n  return 0xABAE75\n  } else if  (atom.serial == 229){\n  return 0xABAE75\n  } else if  (atom.serial == 230){\n  return 0xABAE75\n  } else if  (atom.serial == 231){\n  return 0xABAE75\n  } else if  (atom.serial == 232){\n  return 0xB4AF7A\n  } else if  (atom.serial == 233){\n  return 0xBDB180\n  } else if  (atom.serial == 234){\n  return 0xBDB180\n  } else if  (atom.serial == 235){\n  return 0xBDB180\n  } else if  (atom.serial == 236){\n  return 0xBDB180\n  } else if  (atom.serial == 237){\n  return 0xB9B07D\n  } else if  (atom.serial == 238){\n  return 0xB5B07B\n  } else if  (atom.serial == 239){\n  return 0xB5B07B\n  } else if  (atom.serial == 240){\n  return 0xB5B07B\n  } else if  (atom.serial == 241){\n  return 0xB5B07B\n  } else if  (atom.serial == 242){\n  return 0xB1AF79\n  } else if  (atom.serial == 243){\n  return 0xAEAF77\n  } else if  (atom.serial == 244){\n  return 0xAEAF77\n  } else if  (atom.serial == 245){\n  return 0xAEAF77\n  } else if  (atom.serial == 246){\n  return 0xAEAF77\n  } else if  (atom.serial == 247){\n  return 0xB4AF7A\n  } else if  (atom.serial == 248){\n  return 0xB9B07D\n  } else if  (atom.serial == 249){\n  return 0xB9B07D\n  } else if  (atom.serial == 250){\n  return 0xB9B07D\n  } else if  (atom.serial == 251){\n  return 0xB9B07D\n  } else if  (atom.serial == 252){\n  return 0xB6B07C\n  } else if  (atom.serial == 253){\n  return 0xB3AF7A\n  } else if  (atom.serial == 254){\n  return 0xB3AF7A\n  } else if  (atom.serial == 255){\n  return 0xB3AF7A\n  } else if  (atom.serial == 256){\n  return 0xB3AF7A\n  } else if  (atom.serial == 257){\n  return 0xA6AE72\n  } else if  (atom.serial == 258){\n  return 0x99AC6B\n  } else if  (atom.serial == 259){\n  return 0x99AC6B\n  } else if  (atom.serial == 260){\n  return 0x99AC6B\n  } else if  (atom.serial == 261){\n  return 0x99AC6B\n  } else if  (atom.serial == 262){\n  return 0xB5B07B\n  } else if  (atom.serial == 263){\n  return 0xD1B38B\n  } else if  (atom.serial == 264){\n  return 0xD1B38B\n  } else if  (atom.serial == 265){\n  return 0xD1B38B\n  } else if  (atom.serial == 266){\n  return 0xD1B38B\n  } else if  (atom.serial == 267){\n  return 0xABAE75\n  } else if  (atom.serial == 268){\n  return 0x86AA60\n  } else if  (atom.serial == 269){\n  return 0x86AA60\n  } else if  (atom.serial == 270){\n  return 0x86AA60\n  } else if  (atom.serial == 271){\n  return 0x86AA60\n  } else if  (atom.serial == 272){\n  return 0x88AA61\n  } else if  (atom.serial == 273){\n  return 0x89AA62\n  } else if  (atom.serial == 274){\n  return 0x89AA62\n  } else if  (atom.serial == 275){\n  return 0x89AA62\n  } else if  (atom.serial == 276){\n  return 0x89AA62\n  } else if  (atom.serial == 277){\n  return 0xA0AD6F\n  } else if  (atom.serial == 278){\n  return 0xB6B07B\n  } else if  (atom.serial == 279){\n  return 0xB6B07B\n  } else if  (atom.serial == 280){\n  return 0xB6B07B\n  } else if  (atom.serial == 281){\n  return 0xB6B07B\n  } else if  (atom.serial == 282){\n  return 0xBCB07F\n  } else if  (atom.serial == 283){\n  return 0xC1B182\n  } else if  (atom.serial == 284){\n  return 0xC1B182\n  } else if  (atom.serial == 285){\n  return 0xC1B182\n  } else if  (atom.serial == 286){\n  return 0xC1B182\n  } else if  (atom.serial == 287){\n  return 0xC6B285\n  } else if  (atom.serial == 288){\n  return 0xCAB287\n  } else if  (atom.serial == 289){\n  return 0xCAB287\n  } else if  (atom.serial == 290){\n  return 0xCAB287\n  } else if  (atom.serial == 291){\n  return 0xCAB287\n  } else if  (atom.serial == 292){\n  return 0xC3B183\n  } else if  (atom.serial == 293){\n  return 0xBDB07F\n  } else if  (atom.serial == 294){\n  return 0xBDB07F\n  } else if  (atom.serial == 295){\n  return 0xBDB07F\n  } else if  (atom.serial == 296){\n  return 0xBDB07F\n  } else if  (atom.serial == 297){\n  return 0xC4B184\n  } else if  (atom.serial == 298){\n  return 0xCCB288\n  } else if  (atom.serial == 299){\n  return 0xCCB288\n  } else if  (atom.serial == 300){\n  return 0xCCB288\n  } else if  (atom.serial == 301){\n  return 0xCCB288\n  } else if  (atom.serial == 302){\n  return 0xB3AF7A\n  } else if  (atom.serial == 303){\n  return 0x9BAC6C\n  } else if  (atom.serial == 304){\n  return 0x9BAC6C\n  } else if  (atom.serial == 305){\n  return 0x9BAC6C\n  } else if  (atom.serial == 306){\n  return 0x9BAC6C\n  } else if  (atom.serial == 307){\n  return 0xB1AF78\n  } else if  (atom.serial == 308){\n  return 0xC7B285\n  } else if  (atom.serial == 309){\n  return 0xC7B285\n  } else if  (atom.serial == 310){\n  return 0xC7B285\n  } else if  (atom.serial == 311){\n  return 0xC7B285\n  } else if  (atom.serial == 312){\n  return 0xB5B07B\n  } else if  (atom.serial == 313){\n  return 0xA3AD71\n  } else if  (atom.serial == 314){\n  return 0xA3AD71\n  } else if  (atom.serial == 315){\n  return 0xA3AD71\n  } else if  (atom.serial == 316){\n  return 0xA3AD71\n  } else if  (atom.serial == 317){\n  return 0xAEAF77\n  } else if  (atom.serial == 318){\n  return 0xB9B07D\n  } else if  (atom.serial == 319){\n  return 0xB9B07D\n  } else if  (atom.serial == 320){\n  return 0xB9B07D\n  } else if  (atom.serial == 321){\n  return 0xB9B07D\n  } else if  (atom.serial == 322){\n  return 0xB9B07D\n  } else if  (atom.serial == 327){\n  return 0x8AAB62\n  } else if  (atom.serial == 328){\n  return 0x8AAB62\n  } else if  (atom.serial == 329){\n  return 0x8AAB62\n  } else if  (atom.serial == 330){\n  return 0x8AAB62\n  } else if  (atom.serial == 331){\n  return 0x8AAB62\n  } else if  (atom.serial == 332){\n  return 0x8AAB62\n  } else  {\n  return 0xD1B38B\n  }\n }\n \n            })\n            \n            this._updateId(schemeId, 'color1')\n        "
            ],
            "kwargs": {}
          }
        ],
        "_msg_q": [],
        "_ready": true,
        "layout": "IPY_MODEL_7931c71a4b2449faae9f1a62d61642e9"
      }
    },
    "e215ded4263745cba74f91d7c63fbc35": {
      "model_name": "LayoutModel",
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "state": {}
    },
    "13e570b114f4462dae294964e8d82145": {
      "model_name": "NGLModel",
      "model_module": "nglview-js-widgets",
      "model_module_version": "3.0.1",
      "state": {
        "_camera_orientation": [
          3.659950660761878,
          31.82935062645385,
          -40.07317748585994,
          0,
          13.158076103700079,
          38.239577426979714,
          31.574706735130444,
          0,
          49.455355103004116,
          -12.529545516837235,
          -5.435135208205676,
          0,
          -36.03999996185303,
          -41.74499988555908,
          -40.179999351501465,
          1
        ],
        "_dom_classes": [],
        "_ibtn_fullscreen": "IPY_MODEL_ba02171af9a044809aa7a7ec695e6e80",
        "_igui": null,
        "_iplayer": "IPY_MODEL_30b1d620aec94e228cb970099a5f9e11",
        "_ngl_color_dict": {},
        "_ngl_coordinate_resource": {},
        "_ngl_full_stage_parameters": {
          "impostor": true,
          "quality": "medium",
          "workerDefault": true,
          "sampleLevel": 0,
          "backgroundColor": "white",
          "rotateSpeed": 2,
          "zoomSpeed": 1.2,
          "panSpeed": 1,
          "clipNear": 0,
          "clipFar": 100,
          "clipDist": 10,
          "fogNear": 50,
          "fogFar": 100,
          "cameraFov": 40,
          "cameraEyeSep": 0.3,
          "cameraType": "perspective",
          "lightColor": 14540253,
          "lightIntensity": 1,
          "ambientColor": 14540253,
          "ambientIntensity": 0.2,
          "hoverTimeout": 0,
          "tooltip": true,
          "mousePreset": "default"
        },
        "_ngl_msg_archive": [
          {
            "target": "Stage",
            "type": "call_method",
            "methodName": "loadFile",
            "reconstruc_color_scheme": false,
            "args": [
              {
                "type": "blob",
                "data": "HEADER    \nTITLE     MDANALYSIS FRAME 0: Created by PDBWriter\nCRYST1   77.949   77.949   77.949  90.00  90.00  90.00 P 1           1\nATOM      1  N   ILE B 219      18.290  31.480  31.150  1.00  0.10      B    N\nATOM      2  CA  ILE B 219      18.260  32.500  32.160  1.00  0.10      B    C\nATOM      3  C   ILE B 219      19.450  33.440  31.990  1.00  0.10      B    C\nATOM      4  O   ILE B 219      19.230  34.580  31.780  1.00  0.10      B    O\nATOM      5  N   ASP B 220      20.620  32.920  32.240  1.00  0.10      B    N\nATOM      6  H   ASP B 220      20.780  31.970  32.560  1.00  0.10      B    H\nATOM      7  CA  ASP B 220      21.820  33.740  32.050  1.00  0.10      B    C\nATOM      8  C   ASP B 220      22.970  32.970  31.480  1.00  0.10      B    C\nATOM      9  O   ASP B 220      23.040  31.740  31.490  1.00  0.10      B    O\nATOM     10  N   ALA B 221      23.950  33.700  30.920  1.00  0.10      B    N\nATOM     11  H   ALA B 221      23.860  34.700  30.990  1.00  0.10      B    H\nATOM     12  CA  ALA B 221      24.950  33.150  30.000  1.00  0.10      B    C\nATOM     13  C   ALA B 221      25.920  32.180  30.690  1.00  0.10      B    C\nATOM     14  O   ALA B 221      26.410  32.370  31.800  1.00  0.10      B    O\nATOM     15  N   ILE B 222      26.380  31.160  29.910  1.00  0.10      B    N\nATOM     16  H   ILE B 222      25.800  30.800  29.170  1.00  0.10      B    H\nATOM     17  CA  ILE B 222      27.460  30.270  30.260  1.00  0.10      B    C\nATOM     18  C   ILE B 222      28.820  30.730  29.820  1.00  0.10      B    C\nATOM     19  O   ILE B 222      29.820  30.300  30.380  1.00  0.10      B    O\nATOM     20  N   VAL B 223      28.910  31.670  28.890  1.00  0.10      B    N\nATOM     21  H   VAL B 223      28.060  32.090  28.550  1.00  0.10      B    H\nATOM     22  CA  VAL B 223      30.110  32.310  28.330  1.00  0.10      B    C\nATOM     23  C   VAL B 223      30.520  33.590  29.060  1.00  0.10      B    C\nATOM     24  O   VAL B 223      29.680  34.170  29.720  1.00  0.10      B    O\nATOM     25  N   GLY B 224      31.740  34.100  28.760  1.00  0.10      B    N\nATOM     26  H   GLY B 224      32.360  33.510  28.230  1.00  0.10      B    H\nATOM     27  CA  GLY B 224      32.190  35.440  29.190  1.00  0.10      B    C\nATOM     28  C   GLY B 224      32.020  35.730  30.690  1.00  0.10      B    C\nATOM     29  O   GLY B 224      31.230  36.610  31.140  1.00  0.10      B    O\nATOM     30  N   ARG B 225      32.650  34.860  31.480  1.00  0.10      B    N\nATOM     31  H   ARG B 225      33.240  34.140  31.090  1.00  0.10      B    H\nATOM     32  CA  ARG B 225      32.490  34.910  32.990  1.00  0.17      B    C\nATOM     33  C   ARG B 225      33.650  34.250  33.770  1.00  0.17      B    C\nATOM     34  O   ARG B 225      34.320  33.480  33.200  1.00  0.17      B    O\nATOM     35  N   ASN B 226      33.900  34.760  34.940  1.00  0.17      B    N\nATOM     36  H   ASN B 226      33.360  35.540  35.300  1.00  0.17      B    H\nATOM     37  CA  ASN B 226      34.750  34.130  35.920  1.00  0.18      B    C\nATOM     38  C   ASN B 226      33.920  33.560  37.080  1.00  0.19      B    C\nATOM     39  O   ASN B 226      33.080  34.300  37.590  1.00  0.19      B    O\nATOM     40  N   SER B 227      34.110  32.340  37.570  1.00  0.19      B    N\nATOM     41  H   SER B 227      34.780  31.760  37.090  1.00  0.19      B    H\nATOM     42  CA  SER B 227      33.180  31.590  38.410  1.00  0.17      B    C\nATOM     43  C   SER B 227      33.820  30.720  39.470  1.00  0.15      B    C\nATOM     44  O   SER B 227      34.720  29.910  39.230  1.00  0.15      B    O\nATOM     45  N   ALA B 228      33.310  30.820  40.700  1.00  0.15      B    N\nATOM     46  H   ALA B 228      32.540  31.460  40.840  1.00  0.15      B    H\nATOM     47  CA  ALA B 228      33.600  30.050  41.830  1.00  0.21      B    C\nATOM     48  C   ALA B 228      32.270  29.570  42.450  1.00  0.27      B    C\nATOM     49  O   ALA B 228      31.210  30.250  42.430  1.00  0.27      B    O\nATOM     50  N   LYS B 229      32.400  28.360  42.980  1.00  0.27      B    N\nATOM     51  H   LYS B 229      33.300  27.910  42.900  1.00  0.27      B    H\nATOM     52  CA  LYS B 229      31.340  27.660  43.630  1.00  0.25      B    C\nATOM     53  C   LYS B 229      31.190  28.010  45.090  1.00  0.22      B    C\nATOM     54  O   LYS B 229      30.100  28.310  45.540  1.00  0.22      B    O\nATOM     55  N   ASP B 230      32.290  28.180  45.870  1.00  0.22      B    N\nATOM     56  H   ASP B 230      33.080  27.720  45.430  1.00  0.22      B    H\nATOM     57  CA  ASP B 230      32.370  28.660  47.270  1.00  0.16      B    C\nATOM     58  C   ASP B 230      33.650  29.400  47.700  1.00  0.10      B    C\nATOM     59  O   ASP B 230      34.730  28.830  47.600  1.00  0.10      B    O\nATOM     60  N   ILE B 231      33.580  30.670  47.990  1.00  0.10      B    N\nATOM     61  H   ILE B 231      32.660  31.060  47.820  1.00  0.10      B    H\nATOM     62  CA  ILE B 231      34.730  31.530  48.440  1.00  0.13      B    C\nATOM     63  C   ILE B 231      34.460  31.700  49.960  1.00  0.15      B    C\nATOM     64  O   ILE B 231      33.460  32.350  50.310  1.00  0.15      B    O\nATOM     65  N   ARG B 232      35.350  31.200  50.760  1.00  0.15      B    N\nATOM     66  H   ARG B 232      36.130  30.770  50.280  1.00  0.15      B    H\nATOM     67  CA  ARG B 232      35.290  31.290  52.200  1.00  0.17      B    C\nATOM     68  C   ARG B 232      36.540  31.950  52.770  1.00  0.18      B    C\nATOM     69  O   ARG B 232      37.660  31.440  52.560  1.00  0.18      B    O\nATOM     70  N   THR B 233      36.350  33.150  53.290  1.00  0.18      B    N\nATOM     71  H   THR B 233      35.470  33.560  53.540  1.00  0.18      B    H\nATOM     72  CA  THR B 233      37.530  34.020  53.740  1.00  0.22      B    C\nATOM     73  C   THR B 233      37.410  34.520  55.150  1.00  0.26      B    C\nATOM     74  O   THR B 233      36.250  34.590  55.650  1.00  0.26      B    O\nATOM     75  N   GLU B 234      38.530  34.720  55.840  1.00  0.26      B    N\nATOM     76  H   GLU B 234      39.420  34.590  55.370  1.00  0.26      B    H\nATOM     77  CA  GLU B 234      38.620  34.960  57.300  1.00  0.26      B    C\nATOM     78  C   GLU B 234      39.740  35.860  57.780  1.00  0.27      B    C\nATOM     79  O   GLU B 234      40.740  35.920  57.060  1.00  0.27      B    O\nATOM     80  N   GLU B 235      39.680  36.280  59.000  1.00  0.27      B    N\nATOM     81  H   GLU B 235      38.960  35.990  59.640  1.00  0.27      B    H\nATOM     82  CA  GLU B 235      40.830  37.010  59.550  1.00  0.25      B    C\nATOM     83  C   GLU B 235      41.300  38.210  58.690  1.00  0.23      B    C\nATOM     84  O   GLU B 235      42.460  38.220  58.220  1.00  0.23      B    O\nATOM     85  N   ARG B 236      40.460  39.240  58.570  1.00  0.23      B    N\nATOM     86  H   ARG B 236      39.560  39.180  59.030  1.00  0.23      B    H\nATOM     87  CA  ARG B 236      40.790  40.510  57.780  1.00  0.23      B    C\nATOM     88  C   ARG B 236      41.250  40.300  56.360  1.00  0.23      B    C\nATOM     89  O   ARG B 236      41.770  41.260  55.780  1.00  0.23      B    O\nATOM     90  N   ALA B 237      40.950  39.190  55.640  1.00  0.23      B    N\nATOM     91  H   ALA B 237      40.570  38.440  56.210  1.00  0.23      B    H\nATOM     92  CA  ALA B 237      41.080  38.880  54.220  1.00  0.22      B    C\nATOM     93  C   ALA B 237      40.190  39.780  53.330  1.00  0.21      B    C\nATOM     94  O   ALA B 237      39.040  40.160  53.640  1.00  0.21      B    O\nATOM     95  N   ARG B 238      40.550  39.900  52.000  1.00  0.21      B    N\nATOM     96  H   ARG B 238      41.420  39.470  51.710  1.00  0.21      B    H\nATOM     97  CA  ARG B 238      39.820  40.710  51.050  1.00  0.19      B    C\nATOM     98  C   ARG B 238      39.310  40.000  49.870  1.00  0.17      B    C\nATOM     99  O   ARG B 238      40.100  39.180  49.290  1.00  0.17      B    O\nATOM    100  N   VAL B 239      38.040  40.130  49.460  1.00  0.17      B    N\nATOM    101  H   VAL B 239      37.450  40.540  50.170  1.00  0.17      B    H\nATOM    102  CA  VAL B 239      37.430  39.660  48.210  1.00  0.16      B    C\nATOM    103  C   VAL B 239      36.930  40.750  47.270  1.00  0.15      B    C\nATOM    104  O   VAL B 239      36.330  41.670  47.750  1.00  0.15      B    O\nATOM    105  N   GLN B 240      37.180  40.670  45.960  1.00  0.15      B    N\nATOM    106  H   GLN B 240      37.570  39.800  45.640  1.00  0.15      B    H\nATOM    107  CA  GLN B 240      36.520  41.600  44.970  1.00  0.13      B    C\nATOM    108  C   GLN B 240      35.860  40.800  43.850  1.00  0.12      B    C\nATOM    109  O   GLN B 240      36.470  39.920  43.260  1.00  0.12      B    O\nATOM    110  N   LEU B 241      34.630  41.150  43.500  1.00  0.12      B    N\nATOM    111  H   LEU B 241      34.160  41.770  44.150  1.00  0.12      B    H\nATOM    112  CA  LEU B 241      33.880  40.600  42.310  1.00  0.13      B    C\nATOM    113  C   LEU B 241      33.770  41.730  41.280  1.00  0.15      B    C\nATOM    114  O   LEU B 241      32.940  42.630  41.310  1.00  0.15      B    O\nATOM    115  N   GLY B 242      34.720  41.720  40.390  1.00  0.15      B    N\nATOM    116  H   GLY B 242      35.280  40.910  40.150  1.00  0.15      B    H\nATOM    117  CA  GLY B 242      34.840  42.730  39.340  1.00  0.16      B    C\nATOM    118  C   GLY B 242      36.050  42.690  38.480  1.00  0.17      B    C\nATOM    119  O   GLY B 242      37.140  42.370  39.020  1.00  0.17      B    O\nATOM    120  N   ASN B 243      35.970  43.030  37.170  1.00  0.17      B    N\nATOM    121  H   ASN B 243      35.030  42.990  36.780  1.00  0.17      B    H\nATOM    122  CA  ASN B 243      37.150  43.190  36.310  1.00  0.14      B    C\nATOM    123  C   ASN B 243      38.030  44.420  36.690  1.00  0.11      B    C\nATOM    124  O   ASN B 243      37.530  45.350  37.320  1.00  0.11      B    O\nATOM    125  N   VAL B 244      39.320  44.480  36.290  1.00  0.11      B    N\nATOM    126  H   VAL B 244      39.840  43.670  35.970  1.00  0.11      B    H\nATOM    127  CA  VAL B 244      40.210  45.590  36.700  1.00  0.13      B    C\nATOM    128  C   VAL B 244      40.890  46.220  35.460  1.00  0.14      B    C\nATOM    129  O   VAL B 244      41.410  45.500  34.620  1.00  0.14      B    O\nATOM    130  N   VAL B 245      41.020  47.590  35.510  1.00  0.14      B    N\nATOM    131  H   VAL B 245      40.520  48.120  36.220  1.00  0.14      B    H\nATOM    132  CA  VAL B 245      41.820  48.430  34.560  1.00  0.19      B    C\nATOM    133  C   VAL B 245      42.370  49.650  35.270  1.00  0.25      B    C\nATOM    134  O   VAL B 245      41.830  50.060  36.250  1.00  0.25      B    O\nATOM    135  N   THR B 246      43.420  50.260  34.770  1.00  0.25      B    N\nATOM    136  H   THR B 246      43.870  49.830  33.980  1.00  0.25      B    H\nATOM    137  CA  THR B 246      44.160  51.500  35.280  1.00  0.25      B    C\nATOM    138  C   THR B 246      43.730  52.670  34.410  1.00  0.25      B    C\nATOM    139  O   THR B 246      43.490  52.480  33.180  1.00  0.25      B    O\nATOM    140  N   ALA B 247      43.840  53.910  35.000  1.00  0.25      B    N\nATOM    141  H   ALA B 247      44.080  53.930  35.980  1.00  0.25      B    H\nATOM    142  CA  ALA B 247      43.540  55.220  34.320  1.00  0.18      B    C\nATOM    143  C   ALA B 247      44.490  55.620  33.200  1.00  0.11      B    C\nATOM    144  O   ALA B 247      44.020  56.220  32.250  1.00  0.11      B    O\nATOM    145  N   ALA B 248      45.710  55.140  33.130  1.00  0.11      B    N\nATOM    146  H   ALA B 248      46.200  54.930  33.990  1.00  0.11      B    H\nATOM    147  CA  ALA B 248      46.460  55.240  31.850  1.00  0.31      B    C\nATOM    148  C   ALA B 248      46.040  54.250  30.740  1.00  0.50      B    C\nATOM    149  O   ALA B 248      46.180  54.590  29.560  1.00  0.50      B    O\nATOM    150  N   ALA B 249      45.640  53.040  31.060  1.00  0.50      B    N\nATOM    151  H   ALA B 249      45.460  52.890  32.040  1.00  0.50      B    H\nATOM    152  CA  ALA B 249      45.170  52.060  30.120  1.00  0.50      B    C\nATOM    153  C   ALA B 249      43.690  52.290  29.730  1.00  0.10      B    C\nATOM    154  O   ALA B 249      43.150  51.880  28.710  1.00  0.10      B    O\nATOM    155  N   LEU B 250      42.990  53.130  30.500  1.00  0.10      B    N\nATOM    156  H   LEU B 250      43.360  53.450  31.380  1.00  0.10      B    H\nATOM    157  CA  LEU B 250      41.650  53.680  30.070  1.00  0.10      B    C\nATOM    158  C   LEU B 250      41.670  54.870  29.060  1.00  0.10      B    C\nATOM    159  O   LEU B 250      40.670  55.050  28.370  1.00  0.10      B    O\nATOM    160  N   HID B 251      42.750  55.640  29.000  1.00  0.10      B    N\nATOM    161  H   HID B 251      43.460  55.400  29.690  1.00  0.10      B    H\nATOM    162  CA  HID B 251      42.910  56.800  28.130  1.00  0.10      B    C\nATOM    163  C   HID B 251      42.750  56.480  26.650  1.00  0.10      B    C\nATOM    164  O   HID B 251      43.010  55.370  26.140  1.00  0.10      B    O\nATOM    165  N   GLY B 252      42.250  57.460  25.900  1.00  0.10      B    N\nATOM    166  H   GLY B 252      42.290  58.390  26.290  1.00  0.10      B    H\nATOM    167  CA  GLY B 252      41.880  57.380  24.520  1.00  0.10      B    C\nATOM    168  C   GLY B 252      40.390  56.950  24.250  1.00  0.10      B    C\nATOM    169  O   GLY B 252      39.520  57.140  25.090  1.00  0.10      B    O\nATOM    170  N   GLY B 253      40.060  56.520  23.040  1.00  0.10      B    N\nATOM    171  H   GLY B 253      40.740  56.410  22.300  1.00  0.10      B    H\nATOM    172  CA  GLY B 253      38.670  56.230  22.670  1.00  0.10      B    C\nATOM    173  C   GLY B 253      38.030  54.960  23.370  1.00  0.10      B    C\nATOM    174  O   GLY B 253      38.480  54.530  24.480  1.00  0.10      B    O\nATOM    175  N   ILE B 254      36.950  54.500  22.760  1.00  0.10      B    N\nATOM    176  H   ILE B 254      36.700  54.940  21.890  1.00  0.10      B    H\nATOM    177  CA  ILE B 254      36.110  53.350  23.200  1.00  0.10      B    C\nATOM    178  C   ILE B 254      36.660  52.090  22.520  1.00  0.10      B    C\nATOM    179  O   ILE B 254      36.770  52.100  21.280  1.00  0.10      B    O\nATOM    180  N   ARG B 255      37.160  51.070  23.240  1.00  0.10      B    N\nATOM    181  H   ARG B 255      37.020  51.130  24.240  1.00  0.10      B    H\nATOM    182  CA  ARG B 255      37.940  49.950  22.710  1.00  0.10      B    C\nATOM    183  C   ARG B 255      37.070  48.710  22.510  1.00  0.10      B    C\nATOM    184  O   ARG B 255      36.370  48.290  23.450  1.00  0.10      B    O\nATOM    185  N   ILE B 256      37.260  47.990  21.400  1.00  0.10      B    N\nATOM    186  H   ILE B 256      37.960  48.260  20.720  1.00  0.10      B    H\nATOM    187  CA  ILE B 256      36.800  46.640  21.280  1.00  0.10      B    C\nATOM    188  C   ILE B 256      37.750  45.630  21.960  1.00  0.10      B    C\nATOM    189  O   ILE B 256      39.000  45.780  21.830  1.00  0.10      B    O\nATOM    190  N   SER B 257      37.160  44.670  22.700  1.00  0.10      B    N\nATOM    191  H   SER B 257      36.160  44.660  22.810  1.00  0.10      B    H\nATOM    192  CA  SER B 257      37.910  43.780  23.540  1.00  0.10      B    C\nATOM    193  C   SER B 257      37.310  42.310  23.490  1.00  0.10      B    C\nATOM    194  O   SER B 257      36.220  42.090  23.080  1.00  0.10      B    O\nATOM    195  N   ASP B 258      38.130  41.360  23.850  1.00  0.10      B    N\nATOM    196  H   ASP B 258      38.970  41.600  24.360  1.00  0.10      B    H\nATOM    197  CA  ASP B 258      37.770  39.920  23.840  1.00  0.10      B    C\nATOM    198  C   ASP B 258      37.280  39.420  25.170  1.00  0.10      B    C\nATOM    199  O   ASP B 258      37.150  40.150  26.110  1.00  0.10      B    O\nATOM    200  N   GLN B 259      36.950  38.140  25.250  1.00  0.10      B    N\nATOM    201  H   GLN B 259      37.040  37.480  24.490  1.00  0.10      B    H\nATOM    202  CA  GLN B 259      36.410  37.670  26.470  1.00  0.10      B    C\nATOM    203  C   GLN B 259      36.890  36.210  26.730  1.00  0.10      B    C\nATOM    204  O   GLN B 259      37.270  35.490  25.760  1.00  0.10      B    O\nATOM    205  N   THR B 260      36.690  35.650  27.960  1.00  0.10      B    N\nATOM    206  H   THR B 260      36.110  36.200  28.580  1.00  0.10      B    H\nATOM    207  CA  THR B 260      36.990  34.230  28.330  1.00  0.36      B    C\nATOM    208  C   THR B 260      36.020  33.690  29.340  1.00  0.36      B    C\nATOM    209  O   THR B 260      35.430  34.520  30.100  1.00  0.36      B    O\nATOM    210  N   THR B 261      35.950  32.390  29.440  1.00  0.36      B    N\nATOM    211  H   THR B 261      36.410  31.880  28.700  1.00  0.36      B    H\nATOM    212  CA  THR B 261      35.330  31.630  30.450  1.00  0.29      B    C\nATOM    213  C   THR B 261      36.330  30.830  31.320  1.00  0.22      B    C\nATOM    214  O   THR B 261      37.250  30.220  30.730  1.00  0.22      B    O\nATOM    215  N   ASN B 262      36.260  30.880  32.660  1.00  0.22      B    N\nATOM    216  H   ASN B 262      35.610  31.580  32.990  1.00  0.22      B    H\nATOM    217  CA  ASN B 262      37.250  30.530  33.630  1.00  0.16      B    C\nATOM    218  C   ASN B 262      36.560  30.210  34.930  1.00  0.10      B    C\nATOM    219  O   ASN B 262      35.670  30.960  35.390  1.00  0.10      B    O\nATOM    220  N   SER B 263      36.890  29.060  35.540  1.00  0.10      B    N\nATOM    221  H   SER B 263      37.690  28.540  35.200  1.00  0.10      B    H\nATOM    222  CA  SER B 263      36.040  28.340  36.510  1.00  0.12      B    C\nATOM    223  C   SER B 263      36.800  27.560  37.560  1.00  0.15      B    C\nATOM    224  O   SER B 263      37.960  27.110  37.270  1.00  0.15      B    O\nATOM    225  N   VAL B 264      36.270  27.470  38.760  1.00  0.15      B    N\nATOM    226  H   VAL B 264      35.300  27.750  38.780  1.00  0.15      B    H\nATOM    227  CA  VAL B 264      36.860  26.820  39.930  1.00  0.17      B    C\nATOM    228  C   VAL B 264      35.730  26.470  40.910  1.00  0.19      B    C\nATOM    229  O   VAL B 264      34.690  27.060  40.910  1.00  0.19      B    O\nATOM    230  N   GLU B 265      36.030  25.500  41.680  1.00  0.19      B    N\nATOM    231  H   GLU B 265      36.980  25.140  41.650  1.00  0.19      B    H\nATOM    232  CA  GLU B 265      35.140  25.100  42.740  1.00  0.17      B    C\nATOM    233  C   GLU B 265      35.330  26.060  44.000  1.00  0.15      B    C\nATOM    234  O   GLU B 265      34.450  26.890  44.210  1.00  0.15      B    O\nATOM    235  N   THR B 266      36.330  25.800  44.830  1.00  0.15      B    N\nATOM    236  H   THR B 266      37.010  25.100  44.540  1.00  0.15      B    H\nATOM    237  CA  THR B 266      36.540  26.330  46.130  1.00  0.16      B    C\nATOM    238  C   THR B 266      37.670  27.340  46.120  1.00  0.17      B    C\nATOM    239  O   THR B 266      38.750  27.010  45.680  1.00  0.17      B    O\nATOM    240  N   VAL B 267      37.540  28.460  46.820  1.00  0.17      B    N\nATOM    241  H   VAL B 267      36.640  28.590  47.260  1.00  0.17      B    H\nATOM    242  CA  VAL B 267      38.630  29.390  47.170  1.00  0.18      B    C\nATOM    243  C   VAL B 267      38.600  29.680  48.650  1.00  0.18      B    C\nATOM    244  O   VAL B 267      37.500  30.120  49.190  1.00  0.18      B    O\nATOM    245  N   VAL B 268      39.720  29.490  49.390  1.00  0.18      B    N\nATOM    246  H   VAL B 268      40.540  29.060  48.990  1.00  0.18      B    H\nATOM    247  CA  VAL B 268      39.900  29.800  50.830  1.00  0.17      B    C\nATOM    248  C   VAL B 268      41.150  30.630  50.990  1.00  0.16      B    C\nATOM    249  O   VAL B 268      42.160  30.470  50.350  1.00  0.16      B    O\nATOM    250  N   GLY B 269      41.070  31.630  51.900  1.00  0.16      B    N\nATOM    251  H   GLY B 269      40.210  31.680  52.430  1.00  0.16      B    H\nATOM    252  CA  GLY B 269      42.060  32.650  52.100  1.00  0.16      B    C\nATOM    253  C   GLY B 269      41.890  33.370  53.480  1.00  0.17      B    C\nATOM    254  O   GLY B 269      40.810  33.910  53.750  1.00  0.17      B    O\nATOM    255  N   LYS B 270      43.000  33.410  54.220  1.00  0.17      B    N\nATOM    256  H   LYS B 270      43.790  32.860  53.900  1.00  0.17      B    H\nATOM    257  CA  LYS B 270      43.080  34.020  55.540  1.00  0.20      B    C\nATOM    258  C   LYS B 270      44.310  34.830  55.870  1.00  0.23      B    C\nATOM    259  O   LYS B 270      45.280  34.820  55.210  1.00  0.23      B    O\nATOM    260  N   GLY B 271      44.280  35.610  57.050  1.00  0.23      B    N\nATOM    261  H   GLY B 271      43.420  35.560  57.570  1.00  0.23      B    H\nATOM    262  CA  GLY B 271      45.290  36.460  57.620  1.00  0.17      B    C\nATOM    263  C   GLY B 271      45.620  37.570  56.660  1.00  0.10      B    C\nATOM    264  O   GLY B 271      46.790  37.690  56.180  1.00  0.10      B    O\nATOM    265  N   GLU B 272      44.620  38.330  56.210  1.00  0.10      B    N\nATOM    266  H   GLU B 272      43.730  38.070  56.600  1.00  0.10      B    H\nATOM    267  CA  GLU B 272      44.830  39.480  55.300  1.00  0.19      B    C\nATOM    268  C   GLU B 272      45.260  39.170  53.800  1.00  0.28      B    C\nATOM    269  O   GLU B 272      45.680  40.070  53.080  1.00  0.28      B    O\nATOM    270  N   SER B 273      45.140  37.910  53.460  1.00  0.28      B    N\nATOM    271  H   SER B 273      44.650  37.340  54.130  1.00  0.28      B    H\nATOM    272  CA  SER B 273      45.250  37.500  52.040  1.00  0.28      B    C\nATOM    273  C   SER B 273      44.100  38.120  51.140  1.00  0.27      B    C\nATOM    274  O   SER B 273      43.050  38.500  51.690  1.00  0.27      B    O\nATOM    275  N   ARG B 274      44.370  38.370  49.830  1.00  0.27      B    N\nATOM    276  H   ARG B 274      45.300  38.170  49.480  1.00  0.27      B    H\nATOM    277  CA  ARG B 274      43.430  38.980  48.840  1.00  0.22      B    C\nATOM    278  C   ARG B 274      42.970  38.000  47.710  1.00  0.17      B    C\nATOM    279  O   ARG B 274      43.750  37.340  47.040  1.00  0.17      B    O\nATOM    280  N   VAL B 275      41.730  38.010  47.400  1.00  0.17      B    N\nATOM    281  H   VAL B 275      41.210  38.580  48.050  1.00  0.17      B    H\nATOM    282  CA  VAL B 275      41.020  37.280  46.370  1.00  0.15      B    C\nATOM    283  C   VAL B 275      40.350  38.300  45.400  1.00  0.14      B    C\nATOM    284  O   VAL B 275      39.800  39.300  45.730  1.00  0.14      B    O\nATOM    285  N   LEU B 276      40.320  37.940  44.140  1.00  0.14      B    N\nATOM    286  H   LEU B 276      40.860  37.100  43.920  1.00  0.14      B    H\nATOM    287  CA  LEU B 276      39.830  38.780  43.080  1.00  0.13      B    C\nATOM    288  C   LEU B 276      39.270  37.890  42.000  1.00  0.12      B    C\nATOM    289  O   LEU B 276      39.920  37.020  41.400  1.00  0.12      B    O\nATOM    290  N   ILE B 277      37.980  38.020  41.760  1.00  0.12      B    N\nATOM    291  H   ILE B 277      37.490  38.810  42.150  1.00  0.12      B    H\nATOM    292  CA  ILE B 277      37.200  37.270  40.760  1.00  0.13      B    C\nATOM    293  C   ILE B 277      36.940  38.310  39.700  1.00  0.15      B    C\nATOM    294  O   ILE B 277      36.070  39.200  39.800  1.00  0.15      B    O\nATOM    295  N   GLY B 278      37.840  38.350  38.680  1.00  0.15      B    N\nATOM    296  H   GLY B 278      38.440  37.560  38.480  1.00  0.15      B    H\nATOM    297  CA  GLY B 278      37.880  39.420  37.710  1.00  0.13      B    C\nATOM    298  C   GLY B 278      39.080  39.240  36.760  1.00  0.11      B    C\nATOM    299  O   GLY B 278      40.110  38.760  37.200  1.00  0.11      B    O\nATOM    300  N   ASN B 279      38.940  39.650  35.520  1.00  0.11      B    N\nATOM    301  H   ASN B 279      38.130  40.210  35.280  1.00  0.11      B    H\nATOM    302  CA  ASN B 279      40.050  39.760  34.490  1.00  0.17      B    C\nATOM    303  C   ASN B 279      40.860  41.040  34.660  1.00  0.23      B    C\nATOM    304  O   ASN B 279      40.420  41.960  35.360  1.00  0.23      B    O\nATOM    305  N   GLU B 280      42.080  41.210  34.120  1.00  0.23      B    N\nATOM    306  H   GLU B 280      42.310  40.600  33.350  1.00  0.23      B    H\nATOM    307  CA  GLU B 280      42.740  42.550  34.070  1.00  0.18      B    C\nATOM    308  C   GLU B 280      42.900  42.960  32.560  1.00  0.12      B    C\nATOM    309  O   GLU B 280      43.040  42.110  31.670  1.00  0.12      B    O\nATOM    310  N   TYR B 281      42.800  44.310  32.260  1.00  0.12      B    N\nATOM    311  H   TYR B 281      42.620  44.930  33.040  1.00  0.12      B    H\nATOM    312  CA  TYR B 281      42.930  44.990  30.990  1.00  0.17      B    C\nATOM    313  C   TYR B 281      44.090  46.030  31.090  1.00  0.21      B    C\nATOM    314  O   TYR B 281      44.220  46.690  32.090  1.00  0.21      B    O\nATOM    315  N   GLY B 282      44.960  46.090  30.050  1.00  0.21      B    N\nATOM    316  H   GLY B 282      44.750  45.530  29.230  1.00  0.21      B    H\nATOM    317  CA  GLY B 282      46.290  46.670  30.110  1.00  0.18      B    C\nATOM    318  C   GLY B 282      47.250  46.260  31.190  1.00  0.16      B    C\nATOM    319  O   GLY B 282      47.950  47.100  31.770  1.00  0.16      B    O\nATOM    320  N   GLY B 283      47.200  45.000  31.530  1.00  0.16      B    N\nATOM    321  H   GLY B 283      46.570  44.360  31.070  1.00  0.16      B    H\nATOM    322  CA  GLY B 283      47.940  44.500  32.720  1.00  0.16      B    C\nATOM    323  C   GLY B 283      49.460  44.510  32.500  1.00  0.10      B    C\nATOM    324  O   GLY B 283      49.940  44.470  31.340  1.00  0.10      B    O\nATOM    325  N   LYS B 284      50.290  44.520  33.620  1.00  0.10      B    N\nATOM    326  H   LYS B 284      49.830  44.560  34.510  1.00  0.10      B    H\nATOM    327  CA  LYS B 284      51.740  44.740  33.580  1.00  0.27      B    C\nATOM    328  C   LYS B 284      52.590  43.740  34.420  1.00  0.27      B    C\nATOM    329  O   LYS B 284      53.820  43.760  34.460  1.00  0.27      B    O\nATOM    330  N   GLY B 285      51.870  43.080  35.320  1.00  0.27      B    N\nATOM    331  H   GLY B 285      50.870  42.950  35.190  1.00  0.27      B    H\nATOM    332  CA  GLY B 285      52.370  42.440  36.530  1.00  0.27      B    C\nATOM    333  C   GLY B 285      51.350  41.630  37.330  1.00  0.10      B    C\nATOM    334  O   GLY B 285      50.180  42.000  37.300  1.00  0.10      B    O\nATOM    335  N   PHE B 286      51.650  40.500  37.950  1.00  0.10      B    N\nATOM    336  H   PHE B 286      52.580  40.120  37.910  1.00  0.10      B    H\nATOM    337  CA  PHE B 286      50.640  39.710  38.710  1.00  0.10      B    C\nATOM    338  C   PHE B 286      49.980  40.530  39.830  1.00  0.10      B    C\nATOM    339  O   PHE B 286      50.730  40.870  40.730  1.00  0.10      B    O\nATOM    340  N   TRP B 287      48.710  40.770  39.870  1.00  0.10      B    N\nATOM    341  H   TRP B 287      48.240  40.290  39.110  1.00  0.10      B    H\nATOM    342  CA  TRP B 287      47.980  41.540  40.850  1.00  0.10      B    C\nATOM    343  C   TRP B 287      48.350  43.000  40.950  1.00  0.10      B    C\nATOM    344  O   TRP B 287      47.520  43.930  40.900  1.00  0.10      B    O\nATOM    345  N   ASP B 288      49.630  43.350  41.000  1.00  0.10      B    N\nATOM    346  H   ASP B 288      50.300  42.590  41.050  1.00  0.10      B    H\nATOM    347  CA  ASP B 288      50.130  44.730  41.090  1.00  0.10      B    C\nATOM    348  C   ASP B 288      51.420  45.040  40.280  1.00  0.10      B    C\nATOM    349  O   ASP B 288      52.260  44.140  40.120  1.00  0.10      B    O\nATOM    350  N   ASN B 289      51.530  46.300  39.780  1.00  0.10      B    N\nATOM    351  H   ASN B 289      50.780  46.920  40.040  1.00  0.10      B    H\nATOM    352  CA  ASN B 289      52.790  46.900  39.210  1.00  0.10      B    C\nATOM    353  C   ASN B 289      52.870  48.410  39.490  1.00  0.10      B    C\nEND\n",
                "binary": false
              }
            ],
            "kwargs": {
              "name": "nglview.adaptor.MDAnalysisTrajectory",
              "defaultRepresentation": false,
              "ext": "pdb"
            }
          },
          {
            "component_index": 0,
            "target": "compList",
            "type": "call_method",
            "methodName": "removeAllRepresentations",
            "reconstruc_color_scheme": false,
            "args": [],
            "kwargs": {}
          },
          {
            "component_index": 0,
            "target": "compList",
            "type": "call_method",
            "methodName": "addRepresentation",
            "reconstruc_color_scheme": false,
            "args": [
              "ball+stick"
            ],
            "kwargs": {
              "sele": ":B.C/0 or :B.CA/0 or :B.H/0 or :B.N/0 or :B.O/0",
              "color": "color1",
              "radius": "bfactor"
            }
          }
        ],
        "_ngl_original_stage_parameters": {
          "impostor": true,
          "quality": "medium",
          "workerDefault": true,
          "sampleLevel": 0,
          "backgroundColor": "white",
          "rotateSpeed": 2,
          "zoomSpeed": 1.2,
          "panSpeed": 1,
          "clipNear": 0,
          "clipFar": 100,
          "clipDist": 10,
          "fogNear": 50,
          "fogFar": 100,
          "cameraFov": 40,
          "cameraEyeSep": 0.3,
          "cameraType": "perspective",
          "lightColor": 14540253,
          "lightIntensity": 1,
          "ambientColor": 14540253,
          "ambientIntensity": 0.2,
          "hoverTimeout": 0,
          "tooltip": true,
          "mousePreset": "default"
        },
        "_ngl_repr_dict": {
          "0": {
            "0": {
              "type": "ball+stick",
              "params": {
                "lazy": false,
                "visible": true,
                "quality": "medium",
                "sphereDetail": 1,
                "radialSegments": 10,
                "openEnded": true,
                "disableImpostor": false,
                "aspectRatio": 2,
                "lineOnly": false,
                "cylinderOnly": false,
                "multipleBond": "off",
                "bondScale": 0.4,
                "bondSpacing": 1,
                "linewidth": 2,
                "radiusType": "bfactor",
                "radiusData": {},
                "radiusSize": 0.15,
                "radiusScale": 1,
                "assembly": "default",
                "defaultAssembly": "",
                "clipNear": 0,
                "clipRadius": 0,
                "clipCenter": {
                  "x": 0,
                  "y": 0,
                  "z": 0
                },
                "flatShaded": false,
                "opacity": 1,
                "depthWrite": true,
                "side": "double",
                "wireframe": false,
                "colorScheme": "color1",
                "colorScale": "",
                "colorReverse": false,
                "colorValue": 9474192,
                "colorMode": "hcl",
                "roughness": 0.4,
                "metalness": 0,
                "diffuse": 16777215,
                "diffuseInterior": false,
                "useInteriorColor": true,
                "interiorColor": 2236962,
                "interiorDarkening": 0,
                "matrix": {
                  "elements": [
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1
                  ]
                },
                "disablePicking": false,
                "sele": ":B.C/0 or :B.CA/0 or :B.H/0 or :B.N/0 or :B.O/0"
              }
            }
          }
        },
        "_ngl_version": "2.0.0-dev.36",
        "_ngl_view_id": [
          "F0C95FE4-73AC-43B6-AA9F-2CCBEB1D7BDD"
        ],
        "_player_dict": {},
        "_scene_position": {},
        "_scene_rotation": {},
        "_synced_model_ids": [],
        "_synced_repr_model_ids": [],
        "layout": "IPY_MODEL_e215ded4263745cba74f91d7c63fbc35",
        "n_components": 1,
        "picked": {}
      }
    },
    "5e0414ffbc7c43b2a2c1111e3bd7bb08": {
      "model_name": "LayoutModel",
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "state": {}
    },
    "421c5a6ff1d34a23a9a1732dae92b00c": {
      "model_name": "ImageModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "_dom_classes": [],
        "layout": "IPY_MODEL_5e0414ffbc7c43b2a2c1111e3bd7bb08",
        "width": "900.0"
      }
    },
    "1c30302015e74e1881035fae52ae347f": {
      "model_name": "LayoutModel",
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "state": {}
    },
    "23cf16568d1e4580ba208d21d0dfc09a": {
      "model_name": "DescriptionStyleModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {}
    },
    "f13e82bcc20049a98a4b058eadf84410": {
      "model_name": "PlayModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "_dom_classes": [],
        "layout": "IPY_MODEL_1c30302015e74e1881035fae52ae347f",
        "max": 0,
        "style": "IPY_MODEL_23cf16568d1e4580ba208d21d0dfc09a"
      }
    },
    "8b45b78d99cf4201bb05e9007780a0c0": {
      "model_name": "LayoutModel",
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "state": {}
    },
    "0a1181f2b68148d8bddc9bd82cf5c3c1": {
      "model_name": "SliderStyleModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {}
    },
    "eef7ffad81954d518f9f4c77563560ca": {
      "model_name": "IntSliderModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "_dom_classes": [],
        "layout": "IPY_MODEL_8b45b78d99cf4201bb05e9007780a0c0",
        "max": 0,
        "style": "IPY_MODEL_0a1181f2b68148d8bddc9bd82cf5c3c1"
      }
    },
    "8a0e0f122d344b1aac9eea3e9e1b8cec": {
      "model_name": "LayoutModel",
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "state": {}
    },
    "30b1d620aec94e228cb970099a5f9e11": {
      "model_name": "HBoxModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "_dom_classes": [],
        "children": [
          "IPY_MODEL_f13e82bcc20049a98a4b058eadf84410",
          "IPY_MODEL_eef7ffad81954d518f9f4c77563560ca"
        ],
        "layout": "IPY_MODEL_8a0e0f122d344b1aac9eea3e9e1b8cec"
      }
    },
    "beeb77720972426faf04678087309d75": {
      "model_name": "LinkModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "source": [
          "IPY_MODEL_f13e82bcc20049a98a4b058eadf84410",
          "value"
        ],
        "target": [
          "IPY_MODEL_eef7ffad81954d518f9f4c77563560ca",
          "value"
        ]
      }
    },
    "62ea227e60c44372833fe0c9edf00b1c": {
      "model_name": "LinkModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "source": [
          "IPY_MODEL_f13e82bcc20049a98a4b058eadf84410",
          "value"
        ],
        "target": [
          "IPY_MODEL_13e570b114f4462dae294964e8d82145",
          "frame"
        ]
      }
    },
    "2972812bb49845c78785b861d102f280": {
      "model_name": "LinkModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "source": [
          "IPY_MODEL_f13e82bcc20049a98a4b058eadf84410",
          "max"
        ],
        "target": [
          "IPY_MODEL_13e570b114f4462dae294964e8d82145",
          "max_frame"
        ]
      }
    },
    "f41efd305c364dbd83314acf66a0da50": {
      "model_name": "LinkModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "source": [
          "IPY_MODEL_eef7ffad81954d518f9f4c77563560ca",
          "max"
        ],
        "target": [
          "IPY_MODEL_13e570b114f4462dae294964e8d82145",
          "max_frame"
        ]
      }
    },
    "a24625b3adbd4eb4995354ce39b90128": {
      "model_name": "LayoutModel",
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "state": {
        "width": "34px"
      }
    },
    "48ab6bc47d3d4670b76cbe9e2c354a0e": {
      "model_name": "ButtonStyleModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {}
    },
    "ba02171af9a044809aa7a7ec695e6e80": {
      "model_name": "ButtonModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "_dom_classes": [],
        "icon": "compress",
        "layout": "IPY_MODEL_a24625b3adbd4eb4995354ce39b90128",
        "style": "IPY_MODEL_48ab6bc47d3d4670b76cbe9e2c354a0e"
      }
    },
    "172adc767f61413ab8f34a2840aad0f2": {
      "model_name": "LayoutModel",
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "state": {}
    },
    "b35e5ca7023b4433956689c86a85c5c7": {
      "model_name": "NGLModel",
      "model_module": "nglview-js-widgets",
      "model_module_version": "3.0.1",
      "state": {
        "_camera_orientation": [
          75.34386423965262,
          0,
          0,
          0,
          0,
          75.34386423965262,
          0,
          0,
          0,
          0,
          75.34386423965262,
          0,
          -36.03999996185303,
          -41.74499988555908,
          -40.179999351501465,
          1
        ],
        "_dom_classes": [],
        "_ibtn_fullscreen": "IPY_MODEL_7dca229ff2184c889d9ce32bc3a06bf4",
        "_igui": null,
        "_iplayer": "IPY_MODEL_fff8c70765484e0a876bfceb955ef76e",
        "_ngl_color_dict": {},
        "_ngl_coordinate_resource": {},
        "_ngl_full_stage_parameters": {
          "impostor": true,
          "quality": "medium",
          "workerDefault": true,
          "sampleLevel": 0,
          "backgroundColor": "white",
          "rotateSpeed": 2,
          "zoomSpeed": 1.2,
          "panSpeed": 1,
          "clipNear": 0,
          "clipFar": 100,
          "clipDist": 10,
          "fogNear": 50,
          "fogFar": 100,
          "cameraFov": 40,
          "cameraEyeSep": 0.3,
          "cameraType": "perspective",
          "lightColor": 14540253,
          "lightIntensity": 1,
          "ambientColor": 14540253,
          "ambientIntensity": 0.2,
          "hoverTimeout": 0,
          "tooltip": true,
          "mousePreset": "default"
        },
        "_ngl_msg_archive": [
          {
            "target": "Stage",
            "type": "call_method",
            "methodName": "loadFile",
            "reconstruc_color_scheme": false,
            "args": [
              {
                "type": "blob",
                "data": "HEADER    \nTITLE     MDANALYSIS FRAME 0: Created by PDBWriter\nCRYST1   77.949   77.949   77.949  90.00  90.00  90.00 P 1           1\nATOM      1  N   ILE B 219      18.290  31.480  31.150  1.00  0.10      B    N\nATOM      2  CA  ILE B 219      18.260  32.500  32.160  1.00  0.10      B    C\nATOM      3  C   ILE B 219      19.450  33.440  31.990  1.00  0.10      B    C\nATOM      4  O   ILE B 219      19.230  34.580  31.780  1.00  0.10      B    O\nATOM      5  N   ASP B 220      20.620  32.920  32.240  1.00  0.10      B    N\nATOM      6  H   ASP B 220      20.780  31.970  32.560  1.00  0.10      B    H\nATOM      7  CA  ASP B 220      21.820  33.740  32.050  1.00  0.10      B    C\nATOM      8  C   ASP B 220      22.970  32.970  31.480  1.00  0.10      B    C\nATOM      9  O   ASP B 220      23.040  31.740  31.490  1.00  0.10      B    O\nATOM     10  N   ALA B 221      23.950  33.700  30.920  1.00  0.10      B    N\nATOM     11  H   ALA B 221      23.860  34.700  30.990  1.00  0.10      B    H\nATOM     12  CA  ALA B 221      24.950  33.150  30.000  1.00  0.10      B    C\nATOM     13  C   ALA B 221      25.920  32.180  30.690  1.00  0.10      B    C\nATOM     14  O   ALA B 221      26.410  32.370  31.800  1.00  0.10      B    O\nATOM     15  N   ILE B 222      26.380  31.160  29.910  1.00  0.10      B    N\nATOM     16  H   ILE B 222      25.800  30.800  29.170  1.00  0.10      B    H\nATOM     17  CA  ILE B 222      27.460  30.270  30.260  1.00  0.10      B    C\nATOM     18  C   ILE B 222      28.820  30.730  29.820  1.00  0.10      B    C\nATOM     19  O   ILE B 222      29.820  30.300  30.380  1.00  0.10      B    O\nATOM     20  N   VAL B 223      28.910  31.670  28.890  1.00  0.10      B    N\nATOM     21  H   VAL B 223      28.060  32.090  28.550  1.00  0.10      B    H\nATOM     22  CA  VAL B 223      30.110  32.310  28.330  1.00  0.10      B    C\nATOM     23  C   VAL B 223      30.520  33.590  29.060  1.00  0.10      B    C\nATOM     24  O   VAL B 223      29.680  34.170  29.720  1.00  0.10      B    O\nATOM     25  N   GLY B 224      31.740  34.100  28.760  1.00  0.10      B    N\nATOM     26  H   GLY B 224      32.360  33.510  28.230  1.00  0.10      B    H\nATOM     27  CA  GLY B 224      32.190  35.440  29.190  1.00  0.10      B    C\nATOM     28  C   GLY B 224      32.020  35.730  30.690  1.00  0.10      B    C\nATOM     29  O   GLY B 224      31.230  36.610  31.140  1.00  0.10      B    O\nATOM     30  N   ARG B 225      32.650  34.860  31.480  1.00  0.10      B    N\nATOM     31  H   ARG B 225      33.240  34.140  31.090  1.00  0.10      B    H\nATOM     32  CA  ARG B 225      32.490  34.910  32.990  1.00  0.17      B    C\nATOM     33  C   ARG B 225      33.650  34.250  33.770  1.00  0.17      B    C\nATOM     34  O   ARG B 225      34.320  33.480  33.200  1.00  0.17      B    O\nATOM     35  N   ASN B 226      33.900  34.760  34.940  1.00  0.17      B    N\nATOM     36  H   ASN B 226      33.360  35.540  35.300  1.00  0.17      B    H\nATOM     37  CA  ASN B 226      34.750  34.130  35.920  1.00  0.18      B    C\nATOM     38  C   ASN B 226      33.920  33.560  37.080  1.00  0.19      B    C\nATOM     39  O   ASN B 226      33.080  34.300  37.590  1.00  0.19      B    O\nATOM     40  N   SER B 227      34.110  32.340  37.570  1.00  0.19      B    N\nATOM     41  H   SER B 227      34.780  31.760  37.090  1.00  0.19      B    H\nATOM     42  CA  SER B 227      33.180  31.590  38.410  1.00  0.17      B    C\nATOM     43  C   SER B 227      33.820  30.720  39.470  1.00  0.15      B    C\nATOM     44  O   SER B 227      34.720  29.910  39.230  1.00  0.15      B    O\nATOM     45  N   ALA B 228      33.310  30.820  40.700  1.00  0.15      B    N\nATOM     46  H   ALA B 228      32.540  31.460  40.840  1.00  0.15      B    H\nATOM     47  CA  ALA B 228      33.600  30.050  41.830  1.00  0.21      B    C\nATOM     48  C   ALA B 228      32.270  29.570  42.450  1.00  0.27      B    C\nATOM     49  O   ALA B 228      31.210  30.250  42.430  1.00  0.27      B    O\nATOM     50  N   LYS B 229      32.400  28.360  42.980  1.00  0.27      B    N\nATOM     51  H   LYS B 229      33.300  27.910  42.900  1.00  0.27      B    H\nATOM     52  CA  LYS B 229      31.340  27.660  43.630  1.00  0.25      B    C\nATOM     53  C   LYS B 229      31.190  28.010  45.090  1.00  0.22      B    C\nATOM     54  O   LYS B 229      30.100  28.310  45.540  1.00  0.22      B    O\nATOM     55  N   ASP B 230      32.290  28.180  45.870  1.00  0.22      B    N\nATOM     56  H   ASP B 230      33.080  27.720  45.430  1.00  0.22      B    H\nATOM     57  CA  ASP B 230      32.370  28.660  47.270  1.00  0.16      B    C\nATOM     58  C   ASP B 230      33.650  29.400  47.700  1.00  0.10      B    C\nATOM     59  O   ASP B 230      34.730  28.830  47.600  1.00  0.10      B    O\nATOM     60  N   ILE B 231      33.580  30.670  47.990  1.00  0.10      B    N\nATOM     61  H   ILE B 231      32.660  31.060  47.820  1.00  0.10      B    H\nATOM     62  CA  ILE B 231      34.730  31.530  48.440  1.00  0.13      B    C\nATOM     63  C   ILE B 231      34.460  31.700  49.960  1.00  0.15      B    C\nATOM     64  O   ILE B 231      33.460  32.350  50.310  1.00  0.15      B    O\nATOM     65  N   ARG B 232      35.350  31.200  50.760  1.00  0.15      B    N\nATOM     66  H   ARG B 232      36.130  30.770  50.280  1.00  0.15      B    H\nATOM     67  CA  ARG B 232      35.290  31.290  52.200  1.00  0.17      B    C\nATOM     68  C   ARG B 232      36.540  31.950  52.770  1.00  0.18      B    C\nATOM     69  O   ARG B 232      37.660  31.440  52.560  1.00  0.18      B    O\nATOM     70  N   THR B 233      36.350  33.150  53.290  1.00  0.18      B    N\nATOM     71  H   THR B 233      35.470  33.560  53.540  1.00  0.18      B    H\nATOM     72  CA  THR B 233      37.530  34.020  53.740  1.00  0.22      B    C\nATOM     73  C   THR B 233      37.410  34.520  55.150  1.00  0.26      B    C\nATOM     74  O   THR B 233      36.250  34.590  55.650  1.00  0.26      B    O\nATOM     75  N   GLU B 234      38.530  34.720  55.840  1.00  0.26      B    N\nATOM     76  H   GLU B 234      39.420  34.590  55.370  1.00  0.26      B    H\nATOM     77  CA  GLU B 234      38.620  34.960  57.300  1.00  0.26      B    C\nATOM     78  C   GLU B 234      39.740  35.860  57.780  1.00  0.27      B    C\nATOM     79  O   GLU B 234      40.740  35.920  57.060  1.00  0.27      B    O\nATOM     80  N   GLU B 235      39.680  36.280  59.000  1.00  0.27      B    N\nATOM     81  H   GLU B 235      38.960  35.990  59.640  1.00  0.27      B    H\nATOM     82  CA  GLU B 235      40.830  37.010  59.550  1.00  0.25      B    C\nATOM     83  C   GLU B 235      41.300  38.210  58.690  1.00  0.23      B    C\nATOM     84  O   GLU B 235      42.460  38.220  58.220  1.00  0.23      B    O\nATOM     85  N   ARG B 236      40.460  39.240  58.570  1.00  0.23      B    N\nATOM     86  H   ARG B 236      39.560  39.180  59.030  1.00  0.23      B    H\nATOM     87  CA  ARG B 236      40.790  40.510  57.780  1.00  0.23      B    C\nATOM     88  C   ARG B 236      41.250  40.300  56.360  1.00  0.23      B    C\nATOM     89  O   ARG B 236      41.770  41.260  55.780  1.00  0.23      B    O\nATOM     90  N   ALA B 237      40.950  39.190  55.640  1.00  0.23      B    N\nATOM     91  H   ALA B 237      40.570  38.440  56.210  1.00  0.23      B    H\nATOM     92  CA  ALA B 237      41.080  38.880  54.220  1.00  0.22      B    C\nATOM     93  C   ALA B 237      40.190  39.780  53.330  1.00  0.21      B    C\nATOM     94  O   ALA B 237      39.040  40.160  53.640  1.00  0.21      B    O\nATOM     95  N   ARG B 238      40.550  39.900  52.000  1.00  0.21      B    N\nATOM     96  H   ARG B 238      41.420  39.470  51.710  1.00  0.21      B    H\nATOM     97  CA  ARG B 238      39.820  40.710  51.050  1.00  0.19      B    C\nATOM     98  C   ARG B 238      39.310  40.000  49.870  1.00  0.17      B    C\nATOM     99  O   ARG B 238      40.100  39.180  49.290  1.00  0.17      B    O\nATOM    100  N   VAL B 239      38.040  40.130  49.460  1.00  0.17      B    N\nATOM    101  H   VAL B 239      37.450  40.540  50.170  1.00  0.17      B    H\nATOM    102  CA  VAL B 239      37.430  39.660  48.210  1.00  0.16      B    C\nATOM    103  C   VAL B 239      36.930  40.750  47.270  1.00  0.15      B    C\nATOM    104  O   VAL B 239      36.330  41.670  47.750  1.00  0.15      B    O\nATOM    105  N   GLN B 240      37.180  40.670  45.960  1.00  0.15      B    N\nATOM    106  H   GLN B 240      37.570  39.800  45.640  1.00  0.15      B    H\nATOM    107  CA  GLN B 240      36.520  41.600  44.970  1.00  0.13      B    C\nATOM    108  C   GLN B 240      35.860  40.800  43.850  1.00  0.12      B    C\nATOM    109  O   GLN B 240      36.470  39.920  43.260  1.00  0.12      B    O\nATOM    110  N   LEU B 241      34.630  41.150  43.500  1.00  0.12      B    N\nATOM    111  H   LEU B 241      34.160  41.770  44.150  1.00  0.12      B    H\nATOM    112  CA  LEU B 241      33.880  40.600  42.310  1.00  0.13      B    C\nATOM    113  C   LEU B 241      33.770  41.730  41.280  1.00  0.15      B    C\nATOM    114  O   LEU B 241      32.940  42.630  41.310  1.00  0.15      B    O\nATOM    115  N   GLY B 242      34.720  41.720  40.390  1.00  0.15      B    N\nATOM    116  H   GLY B 242      35.280  40.910  40.150  1.00  0.15      B    H\nATOM    117  CA  GLY B 242      34.840  42.730  39.340  1.00  0.16      B    C\nATOM    118  C   GLY B 242      36.050  42.690  38.480  1.00  0.17      B    C\nATOM    119  O   GLY B 242      37.140  42.370  39.020  1.00  0.17      B    O\nATOM    120  N   ASN B 243      35.970  43.030  37.170  1.00  0.17      B    N\nATOM    121  H   ASN B 243      35.030  42.990  36.780  1.00  0.17      B    H\nATOM    122  CA  ASN B 243      37.150  43.190  36.310  1.00  0.14      B    C\nATOM    123  C   ASN B 243      38.030  44.420  36.690  1.00  0.11      B    C\nATOM    124  O   ASN B 243      37.530  45.350  37.320  1.00  0.11      B    O\nATOM    125  N   VAL B 244      39.320  44.480  36.290  1.00  0.11      B    N\nATOM    126  H   VAL B 244      39.840  43.670  35.970  1.00  0.11      B    H\nATOM    127  CA  VAL B 244      40.210  45.590  36.700  1.00  0.13      B    C\nATOM    128  C   VAL B 244      40.890  46.220  35.460  1.00  0.14      B    C\nATOM    129  O   VAL B 244      41.410  45.500  34.620  1.00  0.14      B    O\nATOM    130  N   VAL B 245      41.020  47.590  35.510  1.00  0.14      B    N\nATOM    131  H   VAL B 245      40.520  48.120  36.220  1.00  0.14      B    H\nATOM    132  CA  VAL B 245      41.820  48.430  34.560  1.00  0.19      B    C\nATOM    133  C   VAL B 245      42.370  49.650  35.270  1.00  0.25      B    C\nATOM    134  O   VAL B 245      41.830  50.060  36.250  1.00  0.25      B    O\nATOM    135  N   THR B 246      43.420  50.260  34.770  1.00  0.25      B    N\nATOM    136  H   THR B 246      43.870  49.830  33.980  1.00  0.25      B    H\nATOM    137  CA  THR B 246      44.160  51.500  35.280  1.00  0.25      B    C\nATOM    138  C   THR B 246      43.730  52.670  34.410  1.00  0.25      B    C\nATOM    139  O   THR B 246      43.490  52.480  33.180  1.00  0.25      B    O\nATOM    140  N   ALA B 247      43.840  53.910  35.000  1.00  0.25      B    N\nATOM    141  H   ALA B 247      44.080  53.930  35.980  1.00  0.25      B    H\nATOM    142  CA  ALA B 247      43.540  55.220  34.320  1.00  0.18      B    C\nATOM    143  C   ALA B 247      44.490  55.620  33.200  1.00  0.11      B    C\nATOM    144  O   ALA B 247      44.020  56.220  32.250  1.00  0.11      B    O\nATOM    145  N   ALA B 248      45.710  55.140  33.130  1.00  0.11      B    N\nATOM    146  H   ALA B 248      46.200  54.930  33.990  1.00  0.11      B    H\nATOM    147  CA  ALA B 248      46.460  55.240  31.850  1.00  0.31      B    C\nATOM    148  C   ALA B 248      46.040  54.250  30.740  1.00  0.50      B    C\nATOM    149  O   ALA B 248      46.180  54.590  29.560  1.00  0.50      B    O\nATOM    150  N   ALA B 249      45.640  53.040  31.060  1.00  0.50      B    N\nATOM    151  H   ALA B 249      45.460  52.890  32.040  1.00  0.50      B    H\nATOM    152  CA  ALA B 249      45.170  52.060  30.120  1.00  0.50      B    C\nATOM    153  C   ALA B 249      43.690  52.290  29.730  1.00  0.10      B    C\nATOM    154  O   ALA B 249      43.150  51.880  28.710  1.00  0.10      B    O\nATOM    155  N   LEU B 250      42.990  53.130  30.500  1.00  0.10      B    N\nATOM    156  H   LEU B 250      43.360  53.450  31.380  1.00  0.10      B    H\nATOM    157  CA  LEU B 250      41.650  53.680  30.070  1.00  0.10      B    C\nATOM    158  C   LEU B 250      41.670  54.870  29.060  1.00  0.10      B    C\nATOM    159  O   LEU B 250      40.670  55.050  28.370  1.00  0.10      B    O\nATOM    160  N   HID B 251      42.750  55.640  29.000  1.00  0.10      B    N\nATOM    161  H   HID B 251      43.460  55.400  29.690  1.00  0.10      B    H\nATOM    162  CA  HID B 251      42.910  56.800  28.130  1.00  0.10      B    C\nATOM    163  C   HID B 251      42.750  56.480  26.650  1.00  0.10      B    C\nATOM    164  O   HID B 251      43.010  55.370  26.140  1.00  0.10      B    O\nATOM    165  N   GLY B 252      42.250  57.460  25.900  1.00  0.10      B    N\nATOM    166  H   GLY B 252      42.290  58.390  26.290  1.00  0.10      B    H\nATOM    167  CA  GLY B 252      41.880  57.380  24.520  1.00  0.10      B    C\nATOM    168  C   GLY B 252      40.390  56.950  24.250  1.00  0.10      B    C\nATOM    169  O   GLY B 252      39.520  57.140  25.090  1.00  0.10      B    O\nATOM    170  N   GLY B 253      40.060  56.520  23.040  1.00  0.10      B    N\nATOM    171  H   GLY B 253      40.740  56.410  22.300  1.00  0.10      B    H\nATOM    172  CA  GLY B 253      38.670  56.230  22.670  1.00  0.10      B    C\nATOM    173  C   GLY B 253      38.030  54.960  23.370  1.00  0.10      B    C\nATOM    174  O   GLY B 253      38.480  54.530  24.480  1.00  0.10      B    O\nATOM    175  N   ILE B 254      36.950  54.500  22.760  1.00  0.10      B    N\nATOM    176  H   ILE B 254      36.700  54.940  21.890  1.00  0.10      B    H\nATOM    177  CA  ILE B 254      36.110  53.350  23.200  1.00  0.10      B    C\nATOM    178  C   ILE B 254      36.660  52.090  22.520  1.00  0.10      B    C\nATOM    179  O   ILE B 254      36.770  52.100  21.280  1.00  0.10      B    O\nATOM    180  N   ARG B 255      37.160  51.070  23.240  1.00  0.10      B    N\nATOM    181  H   ARG B 255      37.020  51.130  24.240  1.00  0.10      B    H\nATOM    182  CA  ARG B 255      37.940  49.950  22.710  1.00  0.10      B    C\nATOM    183  C   ARG B 255      37.070  48.710  22.510  1.00  0.10      B    C\nATOM    184  O   ARG B 255      36.370  48.290  23.450  1.00  0.10      B    O\nATOM    185  N   ILE B 256      37.260  47.990  21.400  1.00  0.10      B    N\nATOM    186  H   ILE B 256      37.960  48.260  20.720  1.00  0.10      B    H\nATOM    187  CA  ILE B 256      36.800  46.640  21.280  1.00  0.10      B    C\nATOM    188  C   ILE B 256      37.750  45.630  21.960  1.00  0.10      B    C\nATOM    189  O   ILE B 256      39.000  45.780  21.830  1.00  0.10      B    O\nATOM    190  N   SER B 257      37.160  44.670  22.700  1.00  0.10      B    N\nATOM    191  H   SER B 257      36.160  44.660  22.810  1.00  0.10      B    H\nATOM    192  CA  SER B 257      37.910  43.780  23.540  1.00  0.10      B    C\nATOM    193  C   SER B 257      37.310  42.310  23.490  1.00  0.10      B    C\nATOM    194  O   SER B 257      36.220  42.090  23.080  1.00  0.10      B    O\nATOM    195  N   ASP B 258      38.130  41.360  23.850  1.00  0.10      B    N\nATOM    196  H   ASP B 258      38.970  41.600  24.360  1.00  0.10      B    H\nATOM    197  CA  ASP B 258      37.770  39.920  23.840  1.00  0.10      B    C\nATOM    198  C   ASP B 258      37.280  39.420  25.170  1.00  0.10      B    C\nATOM    199  O   ASP B 258      37.150  40.150  26.110  1.00  0.10      B    O\nATOM    200  N   GLN B 259      36.950  38.140  25.250  1.00  0.10      B    N\nATOM    201  H   GLN B 259      37.040  37.480  24.490  1.00  0.10      B    H\nATOM    202  CA  GLN B 259      36.410  37.670  26.470  1.00  0.10      B    C\nATOM    203  C   GLN B 259      36.890  36.210  26.730  1.00  0.10      B    C\nATOM    204  O   GLN B 259      37.270  35.490  25.760  1.00  0.10      B    O\nATOM    205  N   THR B 260      36.690  35.650  27.960  1.00  0.10      B    N\nATOM    206  H   THR B 260      36.110  36.200  28.580  1.00  0.10      B    H\nATOM    207  CA  THR B 260      36.990  34.230  28.330  1.00  0.36      B    C\nATOM    208  C   THR B 260      36.020  33.690  29.340  1.00  0.36      B    C\nATOM    209  O   THR B 260      35.430  34.520  30.100  1.00  0.36      B    O\nATOM    210  N   THR B 261      35.950  32.390  29.440  1.00  0.36      B    N\nATOM    211  H   THR B 261      36.410  31.880  28.700  1.00  0.36      B    H\nATOM    212  CA  THR B 261      35.330  31.630  30.450  1.00  0.29      B    C\nATOM    213  C   THR B 261      36.330  30.830  31.320  1.00  0.22      B    C\nATOM    214  O   THR B 261      37.250  30.220  30.730  1.00  0.22      B    O\nATOM    215  N   ASN B 262      36.260  30.880  32.660  1.00  0.22      B    N\nATOM    216  H   ASN B 262      35.610  31.580  32.990  1.00  0.22      B    H\nATOM    217  CA  ASN B 262      37.250  30.530  33.630  1.00  0.16      B    C\nATOM    218  C   ASN B 262      36.560  30.210  34.930  1.00  0.10      B    C\nATOM    219  O   ASN B 262      35.670  30.960  35.390  1.00  0.10      B    O\nATOM    220  N   SER B 263      36.890  29.060  35.540  1.00  0.10      B    N\nATOM    221  H   SER B 263      37.690  28.540  35.200  1.00  0.10      B    H\nATOM    222  CA  SER B 263      36.040  28.340  36.510  1.00  0.12      B    C\nATOM    223  C   SER B 263      36.800  27.560  37.560  1.00  0.15      B    C\nATOM    224  O   SER B 263      37.960  27.110  37.270  1.00  0.15      B    O\nATOM    225  N   VAL B 264      36.270  27.470  38.760  1.00  0.15      B    N\nATOM    226  H   VAL B 264      35.300  27.750  38.780  1.00  0.15      B    H\nATOM    227  CA  VAL B 264      36.860  26.820  39.930  1.00  0.17      B    C\nATOM    228  C   VAL B 264      35.730  26.470  40.910  1.00  0.19      B    C\nATOM    229  O   VAL B 264      34.690  27.060  40.910  1.00  0.19      B    O\nATOM    230  N   GLU B 265      36.030  25.500  41.680  1.00  0.19      B    N\nATOM    231  H   GLU B 265      36.980  25.140  41.650  1.00  0.19      B    H\nATOM    232  CA  GLU B 265      35.140  25.100  42.740  1.00  0.17      B    C\nATOM    233  C   GLU B 265      35.330  26.060  44.000  1.00  0.15      B    C\nATOM    234  O   GLU B 265      34.450  26.890  44.210  1.00  0.15      B    O\nATOM    235  N   THR B 266      36.330  25.800  44.830  1.00  0.15      B    N\nATOM    236  H   THR B 266      37.010  25.100  44.540  1.00  0.15      B    H\nATOM    237  CA  THR B 266      36.540  26.330  46.130  1.00  0.16      B    C\nATOM    238  C   THR B 266      37.670  27.340  46.120  1.00  0.17      B    C\nATOM    239  O   THR B 266      38.750  27.010  45.680  1.00  0.17      B    O\nATOM    240  N   VAL B 267      37.540  28.460  46.820  1.00  0.17      B    N\nATOM    241  H   VAL B 267      36.640  28.590  47.260  1.00  0.17      B    H\nATOM    242  CA  VAL B 267      38.630  29.390  47.170  1.00  0.18      B    C\nATOM    243  C   VAL B 267      38.600  29.680  48.650  1.00  0.18      B    C\nATOM    244  O   VAL B 267      37.500  30.120  49.190  1.00  0.18      B    O\nATOM    245  N   VAL B 268      39.720  29.490  49.390  1.00  0.18      B    N\nATOM    246  H   VAL B 268      40.540  29.060  48.990  1.00  0.18      B    H\nATOM    247  CA  VAL B 268      39.900  29.800  50.830  1.00  0.17      B    C\nATOM    248  C   VAL B 268      41.150  30.630  50.990  1.00  0.16      B    C\nATOM    249  O   VAL B 268      42.160  30.470  50.350  1.00  0.16      B    O\nATOM    250  N   GLY B 269      41.070  31.630  51.900  1.00  0.16      B    N\nATOM    251  H   GLY B 269      40.210  31.680  52.430  1.00  0.16      B    H\nATOM    252  CA  GLY B 269      42.060  32.650  52.100  1.00  0.16      B    C\nATOM    253  C   GLY B 269      41.890  33.370  53.480  1.00  0.17      B    C\nATOM    254  O   GLY B 269      40.810  33.910  53.750  1.00  0.17      B    O\nATOM    255  N   LYS B 270      43.000  33.410  54.220  1.00  0.17      B    N\nATOM    256  H   LYS B 270      43.790  32.860  53.900  1.00  0.17      B    H\nATOM    257  CA  LYS B 270      43.080  34.020  55.540  1.00  0.20      B    C\nATOM    258  C   LYS B 270      44.310  34.830  55.870  1.00  0.23      B    C\nATOM    259  O   LYS B 270      45.280  34.820  55.210  1.00  0.23      B    O\nATOM    260  N   GLY B 271      44.280  35.610  57.050  1.00  0.23      B    N\nATOM    261  H   GLY B 271      43.420  35.560  57.570  1.00  0.23      B    H\nATOM    262  CA  GLY B 271      45.290  36.460  57.620  1.00  0.17      B    C\nATOM    263  C   GLY B 271      45.620  37.570  56.660  1.00  0.10      B    C\nATOM    264  O   GLY B 271      46.790  37.690  56.180  1.00  0.10      B    O\nATOM    265  N   GLU B 272      44.620  38.330  56.210  1.00  0.10      B    N\nATOM    266  H   GLU B 272      43.730  38.070  56.600  1.00  0.10      B    H\nATOM    267  CA  GLU B 272      44.830  39.480  55.300  1.00  0.19      B    C\nATOM    268  C   GLU B 272      45.260  39.170  53.800  1.00  0.28      B    C\nATOM    269  O   GLU B 272      45.680  40.070  53.080  1.00  0.28      B    O\nATOM    270  N   SER B 273      45.140  37.910  53.460  1.00  0.28      B    N\nATOM    271  H   SER B 273      44.650  37.340  54.130  1.00  0.28      B    H\nATOM    272  CA  SER B 273      45.250  37.500  52.040  1.00  0.28      B    C\nATOM    273  C   SER B 273      44.100  38.120  51.140  1.00  0.27      B    C\nATOM    274  O   SER B 273      43.050  38.500  51.690  1.00  0.27      B    O\nATOM    275  N   ARG B 274      44.370  38.370  49.830  1.00  0.27      B    N\nATOM    276  H   ARG B 274      45.300  38.170  49.480  1.00  0.27      B    H\nATOM    277  CA  ARG B 274      43.430  38.980  48.840  1.00  0.22      B    C\nATOM    278  C   ARG B 274      42.970  38.000  47.710  1.00  0.17      B    C\nATOM    279  O   ARG B 274      43.750  37.340  47.040  1.00  0.17      B    O\nATOM    280  N   VAL B 275      41.730  38.010  47.400  1.00  0.17      B    N\nATOM    281  H   VAL B 275      41.210  38.580  48.050  1.00  0.17      B    H\nATOM    282  CA  VAL B 275      41.020  37.280  46.370  1.00  0.15      B    C\nATOM    283  C   VAL B 275      40.350  38.300  45.400  1.00  0.14      B    C\nATOM    284  O   VAL B 275      39.800  39.300  45.730  1.00  0.14      B    O\nATOM    285  N   LEU B 276      40.320  37.940  44.140  1.00  0.14      B    N\nATOM    286  H   LEU B 276      40.860  37.100  43.920  1.00  0.14      B    H\nATOM    287  CA  LEU B 276      39.830  38.780  43.080  1.00  0.13      B    C\nATOM    288  C   LEU B 276      39.270  37.890  42.000  1.00  0.12      B    C\nATOM    289  O   LEU B 276      39.920  37.020  41.400  1.00  0.12      B    O\nATOM    290  N   ILE B 277      37.980  38.020  41.760  1.00  0.12      B    N\nATOM    291  H   ILE B 277      37.490  38.810  42.150  1.00  0.12      B    H\nATOM    292  CA  ILE B 277      37.200  37.270  40.760  1.00  0.13      B    C\nATOM    293  C   ILE B 277      36.940  38.310  39.700  1.00  0.15      B    C\nATOM    294  O   ILE B 277      36.070  39.200  39.800  1.00  0.15      B    O\nATOM    295  N   GLY B 278      37.840  38.350  38.680  1.00  0.15      B    N\nATOM    296  H   GLY B 278      38.440  37.560  38.480  1.00  0.15      B    H\nATOM    297  CA  GLY B 278      37.880  39.420  37.710  1.00  0.13      B    C\nATOM    298  C   GLY B 278      39.080  39.240  36.760  1.00  0.11      B    C\nATOM    299  O   GLY B 278      40.110  38.760  37.200  1.00  0.11      B    O\nATOM    300  N   ASN B 279      38.940  39.650  35.520  1.00  0.11      B    N\nATOM    301  H   ASN B 279      38.130  40.210  35.280  1.00  0.11      B    H\nATOM    302  CA  ASN B 279      40.050  39.760  34.490  1.00  0.17      B    C\nATOM    303  C   ASN B 279      40.860  41.040  34.660  1.00  0.23      B    C\nATOM    304  O   ASN B 279      40.420  41.960  35.360  1.00  0.23      B    O\nATOM    305  N   GLU B 280      42.080  41.210  34.120  1.00  0.23      B    N\nATOM    306  H   GLU B 280      42.310  40.600  33.350  1.00  0.23      B    H\nATOM    307  CA  GLU B 280      42.740  42.550  34.070  1.00  0.18      B    C\nATOM    308  C   GLU B 280      42.900  42.960  32.560  1.00  0.12      B    C\nATOM    309  O   GLU B 280      43.040  42.110  31.670  1.00  0.12      B    O\nATOM    310  N   TYR B 281      42.800  44.310  32.260  1.00  0.12      B    N\nATOM    311  H   TYR B 281      42.620  44.930  33.040  1.00  0.12      B    H\nATOM    312  CA  TYR B 281      42.930  44.990  30.990  1.00  0.17      B    C\nATOM    313  C   TYR B 281      44.090  46.030  31.090  1.00  0.21      B    C\nATOM    314  O   TYR B 281      44.220  46.690  32.090  1.00  0.21      B    O\nATOM    315  N   GLY B 282      44.960  46.090  30.050  1.00  0.21      B    N\nATOM    316  H   GLY B 282      44.750  45.530  29.230  1.00  0.21      B    H\nATOM    317  CA  GLY B 282      46.290  46.670  30.110  1.00  0.18      B    C\nATOM    318  C   GLY B 282      47.250  46.260  31.190  1.00  0.16      B    C\nATOM    319  O   GLY B 282      47.950  47.100  31.770  1.00  0.16      B    O\nATOM    320  N   GLY B 283      47.200  45.000  31.530  1.00  0.16      B    N\nATOM    321  H   GLY B 283      46.570  44.360  31.070  1.00  0.16      B    H\nATOM    322  CA  GLY B 283      47.940  44.500  32.720  1.00  0.16      B    C\nATOM    323  C   GLY B 283      49.460  44.510  32.500  1.00  0.10      B    C\nATOM    324  O   GLY B 283      49.940  44.470  31.340  1.00  0.10      B    O\nATOM    325  N   LYS B 284      50.290  44.520  33.620  1.00  0.10      B    N\nATOM    326  H   LYS B 284      49.830  44.560  34.510  1.00  0.10      B    H\nATOM    327  CA  LYS B 284      51.740  44.740  33.580  1.00  0.27      B    C\nATOM    328  C   LYS B 284      52.590  43.740  34.420  1.00  0.27      B    C\nATOM    329  O   LYS B 284      53.820  43.760  34.460  1.00  0.27      B    O\nATOM    330  N   GLY B 285      51.870  43.080  35.320  1.00  0.27      B    N\nATOM    331  H   GLY B 285      50.870  42.950  35.190  1.00  0.27      B    H\nATOM    332  CA  GLY B 285      52.370  42.440  36.530  1.00  0.27      B    C\nATOM    333  C   GLY B 285      51.350  41.630  37.330  1.00  0.10      B    C\nATOM    334  O   GLY B 285      50.180  42.000  37.300  1.00  0.10      B    O\nATOM    335  N   PHE B 286      51.650  40.500  37.950  1.00  0.10      B    N\nATOM    336  H   PHE B 286      52.580  40.120  37.910  1.00  0.10      B    H\nATOM    337  CA  PHE B 286      50.640  39.710  38.710  1.00  0.10      B    C\nATOM    338  C   PHE B 286      49.980  40.530  39.830  1.00  0.10      B    C\nATOM    339  O   PHE B 286      50.730  40.870  40.730  1.00  0.10      B    O\nATOM    340  N   TRP B 287      48.710  40.770  39.870  1.00  0.10      B    N\nATOM    341  H   TRP B 287      48.240  40.290  39.110  1.00  0.10      B    H\nATOM    342  CA  TRP B 287      47.980  41.540  40.850  1.00  0.10      B    C\nATOM    343  C   TRP B 287      48.350  43.000  40.950  1.00  0.10      B    C\nATOM    344  O   TRP B 287      47.520  43.930  40.900  1.00  0.10      B    O\nATOM    345  N   ASP B 288      49.630  43.350  41.000  1.00  0.10      B    N\nATOM    346  H   ASP B 288      50.300  42.590  41.050  1.00  0.10      B    H\nATOM    347  CA  ASP B 288      50.130  44.730  41.090  1.00  0.10      B    C\nATOM    348  C   ASP B 288      51.420  45.040  40.280  1.00  0.10      B    C\nATOM    349  O   ASP B 288      52.260  44.140  40.120  1.00  0.10      B    O\nATOM    350  N   ASN B 289      51.530  46.300  39.780  1.00  0.10      B    N\nATOM    351  H   ASN B 289      50.780  46.920  40.040  1.00  0.10      B    H\nATOM    352  CA  ASN B 289      52.790  46.900  39.210  1.00  0.10      B    C\nATOM    353  C   ASN B 289      52.870  48.410  39.490  1.00  0.10      B    C\nEND\n",
                "binary": false
              }
            ],
            "kwargs": {
              "name": "nglview.adaptor.MDAnalysisTrajectory",
              "defaultRepresentation": false,
              "ext": "pdb"
            }
          },
          {
            "component_index": 0,
            "target": "compList",
            "type": "call_method",
            "methodName": "removeAllRepresentations",
            "reconstruc_color_scheme": false,
            "args": [],
            "kwargs": {}
          },
          {
            "component_index": 0,
            "target": "compList",
            "type": "call_method",
            "methodName": "addRepresentation",
            "reconstruc_color_scheme": false,
            "args": [
              "ball+stick"
            ],
            "kwargs": {
              "sele": ":B.C/0 or :B.CA/0 or :B.H/0 or :B.N/0 or :B.O/0",
              "color": "color1",
              "radius": "bfactor"
            }
          }
        ],
        "_ngl_original_stage_parameters": {
          "impostor": true,
          "quality": "medium",
          "workerDefault": true,
          "sampleLevel": 0,
          "backgroundColor": "white",
          "rotateSpeed": 2,
          "zoomSpeed": 1.2,
          "panSpeed": 1,
          "clipNear": 0,
          "clipFar": 100,
          "clipDist": 10,
          "fogNear": 50,
          "fogFar": 100,
          "cameraFov": 40,
          "cameraEyeSep": 0.3,
          "cameraType": "perspective",
          "lightColor": 14540253,
          "lightIntensity": 1,
          "ambientColor": 14540253,
          "ambientIntensity": 0.2,
          "hoverTimeout": 0,
          "tooltip": true,
          "mousePreset": "default"
        },
        "_ngl_repr_dict": {
          "0": {
            "0": {
              "type": "ball+stick",
              "params": {
                "lazy": false,
                "visible": true,
                "quality": "medium",
                "sphereDetail": 1,
                "radialSegments": 10,
                "openEnded": true,
                "disableImpostor": false,
                "aspectRatio": 2,
                "lineOnly": false,
                "cylinderOnly": false,
                "multipleBond": "off",
                "bondScale": 0.4,
                "bondSpacing": 1,
                "linewidth": 2,
                "radiusType": "bfactor",
                "radiusData": {},
                "radiusSize": 0.15,
                "radiusScale": 1,
                "assembly": "default",
                "defaultAssembly": "",
                "clipNear": 0,
                "clipRadius": 0,
                "clipCenter": {
                  "x": 0,
                  "y": 0,
                  "z": 0
                },
                "flatShaded": false,
                "opacity": 1,
                "depthWrite": true,
                "side": "double",
                "wireframe": false,
                "colorScheme": "color1",
                "colorScale": "",
                "colorReverse": false,
                "colorValue": 9474192,
                "colorMode": "hcl",
                "roughness": 0.4,
                "metalness": 0,
                "diffuse": 16777215,
                "diffuseInterior": false,
                "useInteriorColor": true,
                "interiorColor": 2236962,
                "interiorDarkening": 0,
                "matrix": {
                  "elements": [
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1
                  ]
                },
                "disablePicking": false,
                "sele": ":B.C/0 or :B.CA/0 or :B.H/0 or :B.N/0 or :B.O/0"
              }
            }
          }
        },
        "_ngl_serialize": true,
        "_ngl_version": "2.0.0-dev.36",
        "_ngl_view_id": [
          "8D6E8FEC-8CC1-453B-ACB9-BF5EFEE8BC75"
        ],
        "_player_dict": {},
        "_scene_position": {},
        "_scene_rotation": {},
        "_synced_model_ids": [],
        "_synced_repr_model_ids": [],
        "layout": "IPY_MODEL_172adc767f61413ab8f34a2840aad0f2",
        "n_components": 1,
        "picked": {}
      }
    },
    "fc5c34298cbd4ce49fae32032515271e": {
      "model_name": "LayoutModel",
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "state": {}
    },
    "7b7f9bd60ad94998859df8e0d3475c83": {
      "model_name": "ImageModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "_dom_classes": [],
        "layout": "IPY_MODEL_fc5c34298cbd4ce49fae32032515271e",
        "width": "900.0"
      }
    },
    "d57520389a9f4907b34431a41c72e118": {
      "model_name": "LayoutModel",
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "state": {}
    },
    "7fe7f2fc67a34a3d810ebbca154b0fd8": {
      "model_name": "DescriptionStyleModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {}
    },
    "ca2282b3987c42529d3b067756daa412": {
      "model_name": "PlayModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "_dom_classes": [],
        "layout": "IPY_MODEL_d57520389a9f4907b34431a41c72e118",
        "max": 0,
        "style": "IPY_MODEL_7fe7f2fc67a34a3d810ebbca154b0fd8"
      }
    },
    "74b33be117ac4e08a18e7e541747b9a0": {
      "model_name": "LayoutModel",
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "state": {}
    },
    "02be942605ca48b89746621d18ea583a": {
      "model_name": "SliderStyleModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {}
    },
    "d2a635433cf04e09b94fbc8166dc573f": {
      "model_name": "IntSliderModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "_dom_classes": [],
        "layout": "IPY_MODEL_74b33be117ac4e08a18e7e541747b9a0",
        "max": 0,
        "style": "IPY_MODEL_02be942605ca48b89746621d18ea583a"
      }
    },
    "d29829bdaec9445cb4ac6cc699c625ef": {
      "model_name": "LayoutModel",
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "state": {}
    },
    "fff8c70765484e0a876bfceb955ef76e": {
      "model_name": "HBoxModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "_dom_classes": [],
        "children": [
          "IPY_MODEL_ca2282b3987c42529d3b067756daa412",
          "IPY_MODEL_d2a635433cf04e09b94fbc8166dc573f"
        ],
        "layout": "IPY_MODEL_d29829bdaec9445cb4ac6cc699c625ef"
      }
    },
    "0e1126b2460e467aa8be53d29053125d": {
      "model_name": "LinkModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "source": [
          "IPY_MODEL_ca2282b3987c42529d3b067756daa412",
          "value"
        ],
        "target": [
          "IPY_MODEL_d2a635433cf04e09b94fbc8166dc573f",
          "value"
        ]
      }
    },
    "e2ffb03609834376a7e664e048f0d1a8": {
      "model_name": "LinkModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "source": [
          "IPY_MODEL_ca2282b3987c42529d3b067756daa412",
          "value"
        ],
        "target": [
          "IPY_MODEL_b35e5ca7023b4433956689c86a85c5c7",
          "frame"
        ]
      }
    },
    "28401262667f4c2fa3a625a545378e82": {
      "model_name": "LinkModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "source": [
          "IPY_MODEL_ca2282b3987c42529d3b067756daa412",
          "max"
        ],
        "target": [
          "IPY_MODEL_b35e5ca7023b4433956689c86a85c5c7",
          "max_frame"
        ]
      }
    },
    "46dbb82de008412ea352b032f58dd689": {
      "model_name": "LinkModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "source": [
          "IPY_MODEL_d2a635433cf04e09b94fbc8166dc573f",
          "max"
        ],
        "target": [
          "IPY_MODEL_b35e5ca7023b4433956689c86a85c5c7",
          "max_frame"
        ]
      }
    },
    "d603f6fece714876b1bdc8331ea0df87": {
      "model_name": "LayoutModel",
      "model_module": "@jupyter-widgets/base",
      "model_module_version": "1.2.0",
      "state": {
        "width": "34px"
      }
    },
    "f14d5b1eeab9418aa135bafbf752c140": {
      "model_name": "ButtonStyleModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {}
    },
    "7dca229ff2184c889d9ce32bc3a06bf4": {
      "model_name": "ButtonModel",
      "model_module": "@jupyter-widgets/controls",
      "model_module_version": "1.5.0",
      "state": {
        "_dom_classes": [],
        "icon": "compress",
        "layout": "IPY_MODEL_d603f6fece714876b1bdc8331ea0df87",
        "style": "IPY_MODEL_f14d5b1eeab9418aa135bafbf752c140"
      }
    }
  }
}
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "5f6216975e6646e2b04accd91c70446a"}
</script>
<script type="application/vnd.jupyter.widget-view+json">
{"version_major": 2, "version_minor": 0, "model_id": "b35e5ca7023b4433956689c86a85c5c7"}
</script>
