# HDR

Here is [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.463.6496&rep=rep1&type=pdf).

## Instructions (slovenian)
V okviru te naloge implementirajte rekonstrukcijo slike z visokim dinamičnim razponom iz serije slik zajete z kadriranjem izpostavljenosti (ang. exposure bracketing). Naloga je deljena na implementacijo algoritma in aplikacije.

Algoritem pokriva rekonstrukcijo funkcije odziva kamere, dodajanje omejitev na rekonstrukcijo te funkcije ter končno rekonstrukcijo slike scene, kot bi bila zajeta s kamero z visokim dinamičnim razponom (z bitno ločljivostjo večjo od 8 bitov).

V aplikaciji algoritem zapakirate v aplikacijo, ki je sposobna naložiti serijo slik z ustreznimi metapodatki o zajemu, jih poslati skozi algoritem za rekonstrukcijo, ter na koncu prikazati sliko z ustrezno preslikavo za prikaz na zaslonu z nizkim dinamičnim razponom.

Kot predobdelavo za prikaz lahko uporabite algoritem CLAHE. Morate ga sicer za to malo prilagoditi - izenačevanje histograma v osnovi pričakuje, da lahko za sliko izračunamo običajen histogram. V primeru naše rekonstruirane slike to ni preprosto mogoče - vrednosti so običajno tipa float, ki nima preprosto definirane zaloge vrednosti. Zato pri gručenju histograma vrednosti običajno zaokrožimo, pri rekonstrukciji pa bi morali transformacijo izenačevanja interpolirati.

Naloga je vredna 120 točk, ki so dalje razdeljene kot:

- algoritem 60 točk
  - osnovna rekonstrukcija funkcije odziva 20 točk
  - omejitev srednje vrednosti  10 točk
  - omejitev gladkosti 20 točk
  - rekonstrukcija slike 10 točk
- aplikacija 60 točk 
  - nalaganje in obdelava serije slik 20 točk
  - avtomatsko branje časa zajemanja 10 točk
  - prikaz HDR slike 10 točk
  - uporabe CLAHE za predobdelavo prikaza 20 točk

## Prerequisites

### Windows

- Install [CMake](https://cmake.org/download/). We recommend to add CMake to path for easier console using.
- Install [opencv 2.4](https://github.com/opencv/opencv) from sources.
    - Get OpenCV [(github)](https://github.com/opencv/opencv) and put in on C:/ (It can be installed somewhere else, but it's recommended to be close to root dir to avoid too long path error). `git clone https://github.com/opencv/opencv`
    - Checkout on 2.4 branch `git checkout 2.4`.
    - Make build directory .
    - In build directory create project with cmake or cmake-gui (enable `BUILD_EXAMPLES` for later test).
    - Open project in Visual Studio.
    - Build Debug and Release versions.
    - Build `INSTALL` project.
    - Add `opencv_dir/build/bin/Release` and `opencv_dir/build/bin/Debug` to PATH variable. 
    - Test installation by running examples in `opencv/build/install/` dir.
- Install [Eigen3](https://bitbucket.org/eigen/eigen/get/3.3-beta1.zip). 

## Installing
```
git clone https://github.com/nejcgalof/detectorORB.git
```

## Build
You can use cmake-gui or write similar like this:
```
mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" -DOpenCV_DIR="C:/opencv/build" -DEIGEN3_INCLUDE_DIR="c:/eigen" ..
```
