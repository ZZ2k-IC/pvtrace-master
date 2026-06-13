# This Python file uses the following encoding: utf-8
import sys
import os
os.environ['QT_MAC_WANTS_LAYER'] = '1'

from PySide2 import QtWidgets, QtCore
from PySide2.QtWidgets import QApplication, QWidget
from PySide2.QtCore import QFile
from PySide2 import QtGui
from PySide2.QtUiTools import QUiLoader


from pvtrace import *
#from pvtrace.geometry.utils import EPS_ZERO
from pvtrace.light.utils import wavelength_to_rgb
from pvtrace.material.utils import lambertian, isotropic, lambertian_with_fresnel
from pvtrace.light.event import Event
import time
import functools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import trimesh
import pandas as pd
from dataclasses import asdict
import progressbar
import trimesh
from matplotlib.pyplot import plot, hist, scatter
import json

import time



class testingQT(QWidget):
    def __init__(self):
        super(testingQT, self).__init__()
        main_widget = self.load_ui()
        
        self.move(845, 0)
        
        # layout = QtWidgets.QVBoxLayout()
        # layout.addWidget(main_widget)
        # self.setLayout(layout)
        
        self.inputShape = self.findChild(QtWidgets.QComboBox,'comboBox')
        self.STLfile = ''
        self.lumophore = self.findChild(QtWidgets.QComboBox,'comboBox_2')
        self.lumophoreConc = self.findChild(QtWidgets.QLineEdit,'lineEdit_20')
        self.waveguideAbs = self.findChild(QtWidgets.QLineEdit,'lineEdit_23')
        self.waveguideN = self.findChild(QtWidgets.QLineEdit,'lineEdit_24')
        self.lumophorePLQY = self.findChild(QtWidgets.QLineEdit,'lineEdit_25')
        self.dimx = self.findChild(QtWidgets.QLineEdit,'lineEdit')
        self.dimy = self.findChild(QtWidgets.QLineEdit,'lineEdit_13')
        self.dimz = self.findChild(QtWidgets.QLineEdit,'lineEdit_2')
        self.STLfileShow = self.findChild(QtWidgets.QPlainTextEdit,'plainTextEdit')
        self.LumfileShow = self.findChild(QtWidgets.QPlainTextEdit,'plainTextEdit_4')
        self.enclosingBox = self.findChild(QtWidgets.QCheckBox,'checkBox_20')
        self.LSCbounds = np.array([])
        
        self.solarFaceAll = self.findChild(QtWidgets.QCheckBox, 'checkBox')
        self.solarFaceLeft = self.findChild(QtWidgets.QCheckBox,'checkBox_3')
        self.solarFaceRight = self.findChild(QtWidgets.QCheckBox,'checkBox_4')
        self.solarFaceFront = self.findChild(QtWidgets.QCheckBox,'checkBox_5')
        self.solarFaceBack = self.findChild(QtWidgets.QCheckBox,'checkBox_6')
        self.bottomMir = self.findChild(QtWidgets.QCheckBox,'checkBox_2')
        self.bottomScat = self.findChild(QtWidgets.QCheckBox,'checkBox_19')

        self.thinFilm = self.findChild(QtWidgets.QCheckBox,'checkBox_7')
        self.thinFilmThickness = self.findChild(QtWidgets.QLineEdit,'lineEdit_14')
    
        self.LSClayers = self.findChild(QtWidgets.QTabWidget,'tabWidget')
        
        self.lightPattern = self.findChild(QtWidgets.QComboBox,'comboBox_3')
        self.lightDimx = self.findChild(QtWidgets.QLineEdit,'lineEdit_4')
        self.lightDimy = self.findChild(QtWidgets.QLineEdit,'lineEdit_3')
        self.lightWavMin = self.findChild(QtWidgets.QLineEdit,'lineEdit_5')
        self.lightWavMax = self.findChild(QtWidgets.QLineEdit,'lineEdit_6')
        self.lightDiv = self.findChild(QtWidgets.QLineEdit,'lineEdit_7')
        
        self.numRays = self.findChild(QtWidgets.QLineEdit,'lineEdit_8')
        self.wavMin = self.findChild(QtWidgets.QLineEdit,'lineEdit_9')
        self.wavMax = self.findChild(QtWidgets.QLineEdit,'lineEdit_10')
        self.convPlot = self.findChild(QtWidgets.QCheckBox,'checkBox_21')
        self.convThres = self.findChild(QtWidgets.QLineEdit,'lineEdit_21')
        self.showSim = self.findChild(QtWidgets.QCheckBox,'checkBox_22')
        
        self.setSaveFolder = self.findChild(QtWidgets.QToolButton,'toolButton')
        self.saveFolder = ''
        self.saveFolderShow = self.findChild(QtWidgets.QPlainTextEdit,'plainTextEdit_2')
        # self.saveFileNameShow = self.findChild(QtWidgets.QPlainTextEdit,'plainTextEdit_3')
        self.saveFileNameShow = self.findChild(QtWidgets.QLineEdit, 'lineEdit_12')
        self.saveFileName = ''
        self.figDPI = self.findChild(QtWidgets.QLineEdit,'lineEdit_11')
        self.saveInputs = self.findChild(QtWidgets.QPushButton, 'pushButton_2')
        self.saveInputsFile = self.findChild(QtWidgets.QLineEdit,'lineEdit_22')
        self.loadInputs = self.findChild(QtWidgets.QPushButton, 'pushButton_3')

        
        
        # to do
        # absorption/scattering of waveguide
        # backscattering layer
        # add surface scattering to waveguide
        # add additional layered geometries
        
        self.inputShape.currentTextChanged.connect(self.onShapeChanged)
        self.rotateX = False
        self.rotateY = False
        
        self.solarFaceAll.stateChanged.connect(self.onSolarFaceAll)
    
        
        self.lightWavMin.textChanged.connect(self.onLightWavMinChanged)
        self.lightWavMax.textChanged.connect(self.onLightWavMaxChanged)
        
        self.dimx.textChanged.connect(self.onDimXChanged)
        self.dimy.textChanged.connect(self.onDimYChanged)
        
        self.setSaveFolder.clicked.connect(self.onSetSaveFolder)
        
        self.saveInputs.clicked.connect(self.onSaveInputs)
        self.loadInputs.clicked.connect(self.onLoadInputs)
        
        self.finishInput = self.findChild(QtWidgets.QPushButton, 'pushButton')
        self.finishInput.clicked.connect(self.onFinishInputClicked)
        self.thinFilm.clicked.connect(self.onThinFilmClicked)


        # Detector controls
        self.enableDetector = self.findChild(QtWidgets.QCheckBox,'checkBox_detector')
        self.detector_posX = self.findChild(QtWidgets.QLineEdit,'lineEdit_detectorX')
        self.detector_posY = self.findChild(QtWidgets.QLineEdit,'lineEdit_detectorY')
        self.detector_posZ = self.findChild(QtWidgets.QLineEdit,'lineEdit_detectorZ')
        self.detector_direction = self.findChild(QtWidgets.QComboBox,'comboBox_detector_dir')

        # Connect detector signals
        if self.enableDetector is not None:
            self.enableDetector.stateChanged.connect(self.onEnableDetector)

            
        # Second LSC (Waveguide) controls
        self.enableSecondLSC = self.findChild(QtWidgets.QCheckBox,'checkBox_secondLSC')
        self.inputShape2 = self.findChild(QtWidgets.QComboBox,'comboBox_2nd')
        self.STLfile2 = ''
        self.lumophore2 = self.findChild(QtWidgets.QComboBox,'comboBox_2_2nd')
        self.lumophoreConc2 = self.findChild(QtWidgets.QLineEdit,'lineEdit_20_2nd')
        self.waveguideAbs2 = self.findChild(QtWidgets.QLineEdit,'lineEdit_23_2nd')
        self.waveguideN2 = self.findChild(QtWidgets.QLineEdit,'lineEdit_24_2nd')
        self.lumophorePLQY2 = self.findChild(QtWidgets.QLineEdit,'lineEdit_25_2nd')
        self.dimx2 = self.findChild(QtWidgets.QLineEdit,'lineEdit_2nd')
        self.dimy2 = self.findChild(QtWidgets.QLineEdit,'lineEdit_13_2nd')
        self.dimz2 = self.findChild(QtWidgets.QLineEdit,'lineEdit_2_2nd')
        
        # Second LSC positioning
        self.lsc2_offsetX = self.findChild(QtWidgets.QLineEdit,'lineEdit_offsetX')
        self.lsc2_offsetY = self.findChild(QtWidgets.QLineEdit,'lineEdit_offsetY')
        self.lsc2_offsetZ = self.findChild(QtWidgets.QLineEdit,'lineEdit_offsetZ')
        
        # Connect signals
        self.enableSecondLSC.stateChanged.connect(self.onEnableSecondLSC)
        self.inputShape2.currentTextChanged.connect(self.onShape2Changed)


    def onEnableDetector(self):
        """Handle detector enable/disable"""
        if self.enableDetector is None:
            return
            
        enabled = self.enableDetector.isChecked()
        print(f"Detector enabled: {enabled}")
        
        # Enable/disable detector controls
        if self.detector_posX:
            self.detector_posX.setEnabled(enabled)
        if self.detector_posY:
            self.detector_posY.setEnabled(enabled)
        if self.detector_posZ:
            self.detector_posZ.setEnabled(enabled)
        if self.detector_direction:
            self.detector_direction.setEnabled(enabled)
        

    def onEnableSecondLSC(self):
        if self.enableSecondLSC is None:
            print("ERROR: enableSecondLSC is None!")
            return
            
        enabled = self.enableSecondLSC.isChecked()
        print(f"Second LSC enabled: {enabled}")
        
        # Enable/disable the frame containing all second LSC controls
        self.frame_2nd = self.findChild(QtWidgets.QFrame, 'frame_2nd')
        if self.frame_2nd:
            self.frame_2nd.setEnabled(enabled)
            print("Frame enabled/disabled successfully")
        else:
            print("ERROR: frame_2nd not found!")

    def onShape2Changed(self):
        if self.inputShape2 is None:
            print("ERROR: inputShape2 is None!")
            return
            
        if(self.inputShape2.currentText() == 'Import Mesh'):
            # Present choice: Single file or folder with multiple files
            choice = QtWidgets.QMessageBox.question(
                self,
                'Import Mode',
                'Do you want to import multiple STL files from a folder?\n\n'
                'Yes = Select folder containing multiple STL files (recommended)\n'
                'No = Select a single STL file',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes
            )
            
            if choice == QtWidgets.QMessageBox.Yes:
                # Folder mode - select folder
                self.STLfile2 = QtWidgets.QFileDialog.getExistingDirectory(
                    self, 
                    'Select Folder with STL Files'
                )
                if self.STLfile2:
                    print(f"Selected folder: {self.STLfile2}")
                else:
                    print("No folder selected")
            else:
                # Single file mode
                self.STLfile2 = QtWidgets.QFileDialog.getOpenFileName(self, 'Select STL File')
                self.STLfile2 = self.STLfile2[0]
                if self.STLfile2:
                    print(f"Selected STL file: {self.STLfile2}")
                else:
                    print("No file selected")
    
    def load_ui(self):
        loader = QUiLoader()
        path = os.path.join(os.path.dirname(__file__), "form.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        ui = loader.load(ui_file, self)
        ui_file.close()
        
        return ui
    
    def onDimXChanged(self):
        self.lightDimx.setText(self.dimx.text())
        if(self.inputShape.currentText() != 'Import Mesh'):
            self.dimy.setText(self.dimx.text())
            self.lightDimy.setText(self.dimx.text())
    
    def onDimYChanged(self):
        self.lightDimy.setText(self.dimy.text())
        
    def onSolarFaceAll(self):
        allEnabled = self.solarFaceAll.isChecked()
        self.solarFaceLeft.setChecked(allEnabled)
        self.solarFaceRight.setChecked(allEnabled)
        self.solarFaceFront.setChecked(allEnabled)
        self.solarFaceBack.setChecked(allEnabled)
    
    def onThinFilmClicked(self):
        if(self.thinFilm.isChecked()):
            self.thinFilmThickness.setEnabled(True)
        else:
            self.thinFilmThickness.setEnabled(False)
    
    def onLightWavMinChanged(self):
        pass
        # self.wavMin.setText(str(float(self.lightWavMin.text()) - 50))
        self.lightWavMax.setText(str(float(self.lightWavMin.text()) + 2))
        
    def onLightWavMaxChanged(self):
        pass
        # self.wavMax.setText(str(float(self.lightWavMax.text()) + 50))
    
    def onShapeChanged(self):
        if(self.inputShape.currentText() == 'Import Mesh'):
            self.STLfile = QtWidgets.QFileDialog.getOpenFileName(self, 'OpenFile')
            self.STLfile = self.STLfile[0]
            self.STLfileShow.setPlainText(self.STLfile)
            self.mesh = trimesh.load(self.STLfile)
            mesh = self.mesh
            self.LSCdims = mesh.extents
            LSCdims = self.LSCdims
            self.LSCbounds = mesh.bounds
            self.LSCbounds = self.LSCbounds - self.mesh.centroid
            self.dimx.setText("{:.3f}".format(LSCdims[0]))
            self.dimy.setText("{:.3f}".format(LSCdims[1]))
            self.dimz.setText("{:.3f}".format(LSCdims[2]))
            time.sleep(1)
            if(float(self.dimx.text()) < float(self.dimz.text())):
                self.rotateY = True
                self.dimx.setText("{:.3f}".format(LSCdims[2]))
                self.dimz.setText("{:.3f}".format(LSCdims[0]))
            elif(float(self.dimy.text()) < float(self.dimz.text())):
                self.rotateX = True
                self.dimy.setText("{:.3f}".format(LSCdims[2]))
                self.dimz.setText("{:.3f}".format(LSCdims[1]))
        else:
            self.STLfileShow.setPlainText('')
            
    def onSetSaveFolder(self):
        self.saveFolder = QtWidgets.QFileDialog.getExistingDirectory(self, 'OpenFile')
        # self.saveFolder = self.saveFolder[0]
        self.saveFolderShow.setPlainText(self.saveFolder)
        
    def onSaveInputs(self):
        data= {}
        data['LSC'] = []
        data['LSC'].append({
            'shape': self.inputShape.currentText(),
            'STLfile': self.STLfile,
            'dimX': (self.dimx.text()),
            'dimY': (self.dimy.text()),
            'dimZ': (self.dimz.text()),
            'PVedgesLRFB': [self.solarFaceLeft.isChecked(), self.solarFaceRight.isChecked(), self.solarFaceFront.isChecked(), self.solarFaceBack.isChecked()],
            'bottomMir': self.bottomMir.isChecked(),
            'bottomScat': self.bottomScat.isChecked(),
            'thinFilm': self.thinFilm.isChecked(),
            'lumophore': self.lumophore.currentText(),
            'lumophoreConc': (self.lumophoreConc.text()),
            'waveguideAbs': (self.waveguideAbs.text()),
            'lightPattern': self.lightPattern.currentText(),
            'lightDimX': (self.lightDimx.text()),
            'lightDimY': (self.lightDimy.text()),
            'lightWavMin': (self.lightWavMin.text()),
            'lightWavMax': (self.lightWavMax.text()),
            'lightDiv': (self.lightDiv.text()),
            'maxRays': (self.numRays.text()),
            'convThres': (self.convThres.text()),
            'convPlot': self.convPlot.isChecked(),
            'wavMin': (self.wavMin.text()),
            'wavMax': (self.wavMax.text()),
            'enclBox': self.enclosingBox.isChecked(),
            'showSim': self.showSim.isChecked(),
            'saveFolder': self.saveFolder,
            'figDPI': (self.figDPI.text()),
            'resultsFileName': self.saveFileNameShow.text(),
            'inputsFileName': self.saveInputsFile.text(),
             # Second LSC parameters
            'enableSecondLSC': self.enableSecondLSC.isChecked(),
            'shape2': self.inputShape2.currentText() if self.enableSecondLSC.isChecked() else '',
            'STLfile2': self.STLfile2,
            'dimX2': self.dimx2.text() if self.enableSecondLSC.isChecked() else '',
            'dimY2': self.dimy2.text() if self.enableSecondLSC.isChecked() else '',
            'dimZ2': self.dimz2.text() if self.enableSecondLSC.isChecked() else '',
            'lumophore2': self.lumophore2.currentText() if self.enableSecondLSC.isChecked() else '',
            'lumophoreConc2': self.lumophoreConc2.text() if self.enableSecondLSC.isChecked() else '',
            'waveguideAbs2': self.waveguideAbs2.text() if self.enableSecondLSC.isChecked() else '',
            'offsetX': self.lsc2_offsetX.text() if self.enableSecondLSC.isChecked() else '0',
            'offsetY': self.lsc2_offsetY.text() if self.enableSecondLSC.isChecked() else '0',
            'offsetZ': self.lsc2_offsetZ.text() if self.enableSecondLSC.isChecked() else '0'
        })
        folderName = QtWidgets.QFileDialog.getExistingDirectory(self, 'OpenFile')
        fileName = folderName + "/" + self.saveInputsFile.text() + '.txt'
        with open(fileName, 'w') as outfile:
            json.dump(data, outfile)
    
    def onLoadInputs(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')
        fileName = fileName[0]
        with open(fileName) as json_file:
            data = json.load(json_file)
            for p in data['LSC']:
                self.inputShape.setCurrentText(p['shape'])
                self.STLfile = p['STLfile']
                self.STLfileShow.setPlainText(self.STLfile)
                self.dimx.setText(p['dimX'])
                self.dimy.setText(p['dimY'])
                self.dimz.setText(p['dimZ'])
                solarFacesArr = p['PVedgesLRFB']
                self.solarFaceLeft.setChecked(solarFacesArr[0])
                self.solarFaceRight.setChecked(solarFacesArr[1])
                self.solarFaceFront.setChecked(solarFacesArr[2])
                self.solarFaceBack.setChecked(solarFacesArr[3])
                self.bottomMir.setChecked(p['bottomMir'])
                self.bottomScat.setChecked(p['bottomScat'])
                self.thinFilm.setChecked(p['thinFilm'])
                self.lumophore.setCurrentText(p['lumophore'])
                self.lumophoreConc.setText(p['lumophoreConc'])
                try:
                    self.waveguideAbs.setText(p['waveguideAbs'])
                except:
                    print('no waveguide abs')
                try:
                    self.showSim.setText(p['showSim'])
                except:
                    print('no show t/f')
                self.lightPattern.setCurrentText(p['lightPattern'])
                self.lightDimx.setText(p['lightDimX'])
                self.lightDimy.setText(p['lightDimY'])
                self.lightWavMin.setText(p['lightWavMin'])
                self.lightWavMax.setText(p['lightWavMax'])
                self.lightDiv.setText(p['lightDiv'])
                self.numRays.setText(p['maxRays'])
                self.convThres.setText(p['convThres'])
                self.convPlot.setChecked(p['convPlot'])
                self.wavMin.setText(p['wavMin'])
                self.wavMax.setText(p['wavMax'])
                self.enclosingBox.setChecked(p['enclBox'])
                self.saveFolderShow.setPlainText(p['saveFolder'])
                self.saveFolder = p['saveFolder']
                self.figDPI.setText(p['figDPI'])
                self.saveFileNameShow.setText(p['resultsFileName'])
                self.saveFileName = p['resultsFileName']
                self.saveInputsFile.setText(p['inputsFileName'])
        pass
    
    def onFinishInputClicked(self):
        print("LSC Shape: \t\t" + self.inputShape.currentText())
        print("LSC Dimensions:\t\tLength = " + self.dimx.text() + ", Width = " + self.dimy.text() + ", Height = " + self.dimz.text())
        print("Lumophore:\t\t" + self.lumophore.currentText())
        
        print("Light Pattern:\t\t" + self.lightPattern.currentText())
        print("Light Dimensions:\tLength = " + self.lightDimx.text() + ", Width = " + self.lightDimy.text())
        print("Light Wavelengths:\tMin = " + self.lightWavMin.text() + " nm, Max = " + self.lightWavMax.text() + " nm")
        print("Light Divergence:\t" + self.lightDiv.text() + " deg")
        
        print("Num Rays: \t\t" + self.numRays.text())
        print("Wavelength Range:\tMin = " + self.wavMin.text() + " nm, Max = " + self.wavMax.text() + " nm")
        
        dataFile = ''
        if(self.saveFolder != ''):
            self.saveFileName = self.saveFileNameShow.text()
            dataFile = open(self.saveFolder+'/'+self.saveFileName+'.txt','a', encoding='utf-8')

            dataFile.write("LSC Shape\t" + self.inputShape.currentText() + "\n")
            if(self.inputShape.currentText()=='Import Mesh'):
                dataFile.write("LSC STL\t" + self.STLfile + "\n")
            dataFile.write("LSC Length\t" + self.dimx.text() + "\n")
            dataFile.write("LSC Width\t" + self.dimy.text() + "\n")
            dataFile.write("LSC Height\t" + self.dimz.text() + "\n")
            dataFile.write("Lumophore\t" + self.lumophore.currentText() + "\n")
            dataFile.write("Light Pattern\t" + self.lightPattern.currentText() + "\n")
            dataFile.write("Light Length\t" + self.lightDimx.text() + "\n")
            dataFile.write("Light Width\t" + self.lightDimy.text() + "\n")
            dataFile.write("Light Wav Min\t" + self.lightWavMin.text() + "\n")
            dataFile.write("Light Wav Max\t" + self.lightWavMax.text() + "\n")
            dataFile.write("Light Divergence\t" + self.lightDiv.text() + "\n")
            dataFile.write("Num Rays\t" + self.numRays.text() + "\n")
            dataFile.write("Wavelength Range Min\t" + self.wavMin.text() + "\n")
            dataFile.write("Wavelength Range Max\t" + self.wavMax.text() + "\n")
        
        self.entrance_rays, self.exit_rays, self.exit_norms = self.runPVTrace(dataFile)
        if(self.saveFileName != ''):
            dataFile.close()
        # QApplication.quit()
        
    def runPVTrace(self, dataFile):
        print('Input Received')

        # CRITICAL: Reset PVTrace state between runs
        self.resetPVTraceState()

        enableSecondLSC = self.enableSecondLSC.isChecked()
        if enableSecondLSC:
            LSC2dimX = float(self.dimx2.text())
            LSC2dimY = float(self.dimy2.text())
            LSC2dimZ = float(self.dimz2.text())
            LSC2shape = self.inputShape2.currentText()
            LumType2 = self.lumophore2.currentText()
            LumConc2 = float(self.lumophoreConc2.text())
            LumPLQY2 = float(self.lumophorePLQY2.text())
            wavAbs2 = float(self.waveguideAbs2.text())
            wavN2 = float(self.waveguideN2.text())
            
            # Positioning offsets
            offsetX = float(self.lsc2_offsetX.text())
            offsetY = float(self.lsc2_offsetY.text())
            offsetZ = float(self.lsc2_offsetZ.text())

        def createWorld(dim):
            world = Node(
            name="World",
            geometry = Sphere(
                radius = 11*dim,
                material=Material(refractive_index=1.0),
                )   
            )
            
            return world
        
        def createBoxLSC(dimX, dimY, dimZ, wavAbs, wavN):
            LSC = Node(
                name = "LSC",
                geometry = 
                Box(
                    (dimX, dimY, dimZ),
                    material = Material(
                        refractive_index = wavN,
                        components = [
                            Absorber(coefficient = wavAbs*1.0), 
                            Scatterer(coefficient = wavAbs*0.0)
                            ]
                    ),
                ),
                parent = world
            )
            
            return LSC
        
        
        def createCylLSC(dimXY, dimZ, wavAbs, wavN):
            LSC = Node(
                name = "LSC",
                geometry = 
                Cylinder(
                    dimZ, dimXY/2,
                    material = Material(
                        refractive_index = wavN,
                        components = [
                            Absorber(coefficient = wavAbs), 
                            ]
                    ),
                ),
                parent = world
            )
            
            return LSC
        
        
        def createSphLSC(dimXYZ, wavAbs, wavN):
            LSC = Node(
                name = "LSC",
                geometry = 
                Sphere(
                    dimXYZ/2,
                    material = Material(
                        refractive_index = wavN,
                        components = [
                            Absorber(coefficient = wavAbs), 
                            ]
                    ),
                ),
                parent = world
            )
            
            return LSC
        

        def createMeshLSC(self, wavAbs, wavN, stl_file_path=None):
            # Use provided path or default to primary STL file
            if stl_file_path is None:
                stl_file_path = self.STLfile
            
            LSC = Node(
                name = "LSC",
                geometry = 
                Mesh(
                    trimesh = trimesh.load(stl_file_path),  # Use the parameter
                    material = Material(
                        refractive_index = wavN,
                        components = [
                            Absorber(coefficient = wavAbs*1.00), 
                            Scatterer(coefficient = wavAbs*0.00)
                            ]
                    ),
                ),
                parent = world
            )
            LSC.location = [0,0,0]
            return LSC
        
        def createMultiPartMeshLSC(self, wavAbs, wavN, stl_path, parent_node):
            """
            Load multiple STL files from a folder and create separate child nodes.
            This ensures each part has a unique identity for correct container detection.
            
            Args:
                wavAbs: Absorption coefficient
                wavN: Refractive index
                stl_path: Path to either:
                    - A single STL file (backward compatible)
                    - A folder containing multiple STL files
                parent_node: Parent node to attach children to (usually world or LSC)
            
            Returns:
                Tuple of (parent_container, list_of_part_nodes)
            """
            import os
            import glob
            
            # Check if path is a directory or file
            if os.path.isdir(stl_path):
                # FOLDER MODE: Load all STL files from folder
                stl_files = sorted(glob.glob(os.path.join(stl_path, "*.stl")))
                if len(stl_files) == 0:
                    print(f"ERROR: No STL files found in folder: {stl_path}")
                    return None, []
                
                print(f"Folder mode: Found {len(stl_files)} STL file(s) in {stl_path}")
                
                if len(stl_files) == 1:
                    # Single file - create one node
                    mesh = trimesh.load(stl_files[0])
                    print(f"  Loading single file: {os.path.basename(stl_files[0])}")
                    waveguide = Node(
                        name="LSC2_Waveguide",
                        geometry=Mesh(
                            trimesh=mesh,
                            material=Material(
                                refractive_index=wavN,
                                components=[
                                    Absorber(coefficient=wavAbs*1.00),
                                    Scatterer(coefficient=wavAbs*0.00)
                                ]
                            )
                        ),
                        parent=parent_node
                    )
                    return waveguide, [waveguide]
                
                else:
                    # Multiple files - create parent + children
                    print(f"  Creating parent container with {len(stl_files)} child nodes")
                    
                    meshes = []

                    for stl_file in stl_files:
                        mesh = trimesh.load(stl_file)
                        meshes.append(mesh)
                    
                    # Create parent container (no geometry, just for organization)
                    parent_container = Node(
                        name="LSC2_Waveguide_Parent",
                        parent=parent_node
                    )
                    
                    part_nodes = []
                    for i, (stl_file, mesh) in enumerate(zip(stl_files, meshes)):
                        filename = os.path.basename(stl_file)
                        
                        part_node = Node(
                            name=f"LSC2_Waveguide_Part{i}",
                            geometry=Mesh(
                                trimesh=mesh,
                                material=Material(
                                    refractive_index=wavN,
                                    components=[
                                        Absorber(coefficient=wavAbs*1.00),
                                        Scatterer(coefficient=wavAbs*0.00)
                                    ]
                                )
                            ),
                            parent=parent_container
                        )
                        

                        part_node.location = [0.0, 0.0, 0.0]
                        
                        part_nodes.append(part_node)
                        
                        bounds = mesh.bounds
                        print(f"    Part {i} ({filename}):")
                        print(f"      Vertices: {len(mesh.vertices)}")
                        print(f"      Bounds: min=[{bounds[0][0]:.3f}, {bounds[0][1]:.3f}, {bounds[0][2]:.3f}]")
                        print(f"              max=[{bounds[1][0]:.3f}, {bounds[1][1]:.3f}, {bounds[1][2]:.3f}]")
                    
                    return parent_container, part_nodes
            
            else:
                # FILE MODE: Single STL file (backward compatible with old behavior)
                # Try to split into disconnected components
                mesh = trimesh.load(stl_path)
                
                if hasattr(mesh, 'split'):
                    parts = mesh.split()
                else:
                    parts = [mesh]
                
                num_parts = len(parts)
                print(f"Single file mode: Loaded {os.path.basename(stl_path)}")
                print(f"  Found {num_parts} disconnected component(s)")
                
                if num_parts == 1:
                    # Single connected mesh
                    print("  Creating single waveguide node")
                    waveguide = Node(
                        name="LSC2_Waveguide",
                        geometry=Mesh(
                            trimesh=mesh,
                            material=Material(
                                refractive_index=wavN,
                                components=[
                                    Absorber(coefficient=wavAbs*1.00),
                                    Scatterer(coefficient=wavAbs*0.00)
                                ]
                            )
                        ),
                        parent=parent_node
                    )
                    return waveguide, [waveguide]
                
                else:
                    # Multiple disconnected parts in one file
                    print(f"  Creating parent container with {num_parts} child nodes from split")
                    
                    parent_container = Node(
                        name="LSC2_Waveguide_Parent",
                        parent=parent_node
                    )
                    
                    part_nodes = []
                    for i, part in enumerate(parts):
                        part_node = Node(
                            name=f"LSC2_Waveguide_Part{i}",
                            geometry=Mesh(
                                trimesh=part,
                                material=Material(
                                    refractive_index=wavN,
                                    components=[
                                        Absorber(coefficient=wavAbs*1.00),
                                        Scatterer(coefficient=wavAbs*0.00)
                                    ]
                                )
                            ),
                            parent=parent_container
                        )
                        part_nodes.append(part_node)
                        
                        bounds = part.bounds
                        centroid = part.centroid
                        print(f"    Part {i}: vertices={len(part.vertices)}")
                        print(f"      Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
                    
                    return parent_container, part_nodes
            

        def addWaveguideSurfaces(LSC2):
            """Configure surface properties for the waveguide LSC"""
            class WaveguideSurface(FresnelSurfaceDelegate):
                def reflectivity(self, surface, ray, geometry, container, adjacent):
                    # Custom reflectivity for waveguide
                    # You can add total internal reflection logic here
                    return super(WaveguideSurface, self).reflectivity(surface, ray, geometry, container, adjacent)
                
                def transmitted_direction(self, surface, ray, geometry, container, adjacent):
                    # Handle light transmission between LSCs
                    return super(WaveguideSurface, self).transmitted_direction(surface, ray, geometry, container, adjacent)
            
            LSC2.geometry.material.surface = Surface(delegate = WaveguideSurface())
            return LSC2
        
        def addLR305(LSC, LumConc, LumPLQY):
            wavelength_range = (wavMin, wavMax)
            x = np.linspace(wavMin, wavMax, 200)  # wavelength, units: nm
            absorption_spectrum = lumogen_f_red_305.absorption(x)/10*LumConc  # units: mm-1
            emission_spectrum = lumogen_f_red_305.emission(x)/10*LumConc      # units: mm-1
            LSC.geometry.material.components.append(
                Luminophore(
                    coefficient=np.column_stack((x, absorption_spectrum)),
                    emission=np.column_stack((x, emission_spectrum)),
                    quantum_yield=LumPLQY/100,
                    phase_function=isotropic
                    )
                )
            return LSC, x, absorption_spectrum*10/LumConc, emission_spectrum*10/LumConc
        

        def addCeYAG(LSC, LumConc, LumPLQY):
            return None

        def addXYMirror(mirror_z, mirror_x, mirror_y, thickness=1e-3):
            mirror = Node(
                name="xyMirror",
                geometry=Box(
                    (mirror_x, mirror_y, thickness),
                    material=Material(refractive_index=1.0)
                ),
                parent=world
            )
            mirror.location = (0, 0, mirror_z)

            # Remove any absorption/scattering
            mirror.geometry.material.components = []

            class PerfectMirror(FresnelSurfaceDelegate):
                def reflectivity(self, surface, ray, geometry, container, adjacent):
                    return 1.0

            mirror.geometry.material.surface = Surface(delegate=PerfectMirror())
            return mirror

        def addBottomSurf(LSC, bottomMir, bottomScat):
            if(bottomMir or bottomScat):
                bottomSpacer = createBoxLSC(LSCdimX, LSCdimY, LSCdimZ/100)
                bottomSpacer.name = "bottomSpacer"
                bottomSpacer.location=[0,0,-(LSCdimZ + LSCdimZ/100)/2]
                bottomSpacer.geometry.material.refractive_index = 1.0
                del bottomSpacer.geometry.material.components[0]
                
            class BottomReflector(FresnelSurfaceDelegate):
                def reflectivity(self, surface, ray, geometry, container, adjacent):
                    normal = geometry.normal(ray.position)
                    if((bottomMir or bottomScat) and np.allclose(normal, [0,0,-1])):
                        return 1.0
                    
                    return super(BottomReflector, self).reflectivity(surface, ray, geometry, container, adjacent)
                
                def reflected_direction(self, surface, ray, geometry, container, adjacent):
                    normal = geometry.normal(ray.position)
                    if(bottomScat and np.allclose(normal, [0,0,-1])):
                        return tuple(lambertian())
                    return super(BottomReflector, self).reflected_direction(surface, ray, geometry, container, adjacent)
                
                def transmitted_direction(self, surface, ray, geometry, container, adjacent):
                    normal = geometry.normal(ray.position)
                    
                    return super(BottomReflector, self).transmitted_direction(surface, ray, geometry, container, adjacent)
                
            if(bottomMir or bottomScat):
                bottomSpacer.geometry.material.surface = Surface(delegate = BottomReflector())
            
            return LSC
        
        def addSolarCells(LSC, left, right, front, back, allEdges):
            
            class SolarCellEdges(FresnelSurfaceDelegate):
                def reflectivity(self, surface, ray, geometry, container, adjacent):
                    normal = geometry.normal(ray.position)
                    
                    # if(abs(normal[2]- -1)<0.1 and bottom):
                    #     return 1.0
                    
                    # if(allEdges or left or right or front or back == False):
                    #     return super(SolarCellEdges, self).reflectivity(surface, ray, geometry, container, adjacent)
                    
                    # if(abs(normal[0]- -1)<0.1 and left):
                    #     return 0.0
                    # elif(abs(normal[0]- -1)<0.1 and not left):
                    #     return 1.0
                    
                    # if(abs(normal[0]-1)<0.1 and right):
                    #     return 0.0
                    # elif(abs(normal[0]-1)<0.1 and not right):
                    #     return 1.0
                    
                    # if(abs(normal[1]- -1)<0.1 and front):
                    #     return 0.0
                    # elif(abs(normal[1]- -1)<0.1 and not front):
                    #     return 1.0
                    
                    # if(abs(normal[1]-1)<0.1 and back):
                    #     return 0.0
                    # elif(abs(normal[1]-1)<0.1 and not back):
                    #     return 1.0
                    
                    # if(abs(normal[2])<0.2 and allEdges):
                    #     return 0.0
                    
                    
                    if((allEdges or left or right or front or back) == False):
                        return super(SolarCellEdges, self).reflectivity(surface, ray, geometry, container, adjacent)
                    
                    if(abs(normal[0]- -1)<0.1 and left):
                        return 0.0
                    elif(abs(normal[0]- -1)<0.1 and not left):
                        return 1.0
                    
                    if(abs(normal[0]-1)<0.1 and right):
                        return 0.0
                    elif(abs(normal[0]-1)<0.1 and not right):
                        return 1.0
                    
                    if(abs(normal[1]- -1)<0.1 and front):
                        return 0.0
                    elif(abs(normal[1]- -1)<0.1 and not front):
                        return 1.0
                    
                    if(abs(normal[1]-1)<0.1 and back):
                        return 0.0
                    elif(abs(normal[1]-1)<0.1 and not back):
                        return 1.0
                    
                    if(abs(normal[2])<0.2 and allEdges):
                        return 0.0
                    
                    return super(SolarCellEdges, self).reflectivity(surface, ray, geometry, container, adjacent)
                
                def transmitted_direction(self, surface, ray, geometry, container, adjacent):
                    normal = geometry.normal(ray.position)
                    if(abs(normal[0]- -1)<0.1 and left):
                        return ray.position
                    if(abs(normal[0]-1)<0.1 and right):
                        return ray.position
                    if(abs(normal[1]- -1)<0.1 and front):
                        return ray.position
                    if(abs(normal[1]-1)<0.1 and back):
                        return ray.position
                    if(abs(normal[2])<0.2 and allEdges):
                        return ray.position
                    return super(SolarCellEdges, self).transmitted_direction(surface, ray, geometry, container, adjacent)
            
            LSC.geometry.material.surface = Surface(delegate = SolarCellEdges())
            
            return LSC
        
        
        def initLight(lightWavMin, lightWavMax):
            h = 6.626e-34
            c = 3.0e+8
            k = 1.38e-23
            
            def planck(wav, T):
                a = 2.0*h*c**2
                b = h*c/(wav*k*T)
                intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
                return intensity
            
            # generate x-axis in increments from 1nm to 3 micrometer in 1 nm increments
            # starting at 1 nm to avoid wav = 0, which would result in division by zero.
            wavelengths = np.arange(lightWavMin*1e-9, lightWavMax*1e-9, 1e-9)
            intensity5800 = planck(wavelengths, 5800.)
            
            dist = Distribution(wavelengths*1e9, intensity5800)
            
            light = Node(
                name = "Light",
                light = Light(
                    wavelength = lambda: dist.sample(np.random.uniform())
                ),
                parent = world
            )
            
            # Force light source to z=-35
            light.location = (0, 0, 0)
            print(f"Light positioned at: (0, 0, -35)")
            
            return wavelengths*1e9, intensity5800, light
        
        def addRectMask(light, lightDimX, lightDimY):
            light.light.position = functools.partial(rectangular_mask, lightDimX/2, lightDimY/2)
            return light
        
        def addCircMask(light, lightDimX):
            light.light.position = functools.partial(circular_mask, lightDimX/2)
            return light
        
        def addPointSource(light):
            return light
        
        def addStraightRays(light):
            def straight_direction():
                return (0, 0, 1)  # Straight down in -Z direction
            
            light.light.direction = straight_direction
            return light
                
        def addLightDiv(light, lightDiv):
            light.light.direction = functools.partial(isotropic, np.radians(lightDiv))
            return light
        
        def addRealLambertian(light):
            # Assign the function itself (not the result of calling it)
            # Each time light emits, it will call lambertian_with_fresnel() to get a new direction
            light.light.direction = lambda: lambertian_with_fresnel(n1=1.6, n2=1.8)
            return light
        
        # # Add this after loading your direction data (around line 758)
        # direction_data_list = np.load(r"C:\Users\Zedd\OneDrive - Imperial College London\UROP\pvtrace-master\detected_ray_directions_LiNbO3_trapezium.npy")

        # def custom_direction_sampler():
        #     """Sample a random direction from the loaded data"""
        #     if len(direction_data_list) == 0:
        #         # Fallback to default lambertian if no data
        #         return lambertian(np.radians(lightDiv))
            
        #     # Randomly select one direction from the loaded data
        #     random_index = np.random.randint(0, len(direction_data_list))
        #     direction = direction_data_list[random_index]
            
        #     # Normalize the direction vector (ensure it's unit length)
        #     direction = direction / np.linalg.norm(direction)
            
        #     return tuple(direction)
        
        # def addCustomDirection(light):
        #     """Use custom direction sampler for the light source"""
        #     light.light.direction = custom_direction_sampler
        #     return light

        def doRayTracing(numRays, convThres, showSim, use_parallel=True):
            """Modified to support both parallel and sequential processing"""
            
            # For small simulations, use sequential
            if numRays < 200 or not use_parallel:
                return doRayTracingSequential(numRays, convThres, showSim)
            
            # For large simulations, use batch parallel processing
            return doRayTracingBatchParallel(numRays, convThres, showSim)

        def doRayTracingSequential(numRays, convThres, showSim):
            entrance_rays = []
            exit_rays = []
            exit_norms = []
            absorbed_rays = []
            max_rays = numRays
                
            vis = MeshcatRenderer(open_browser=showSim, transparency=False, opacity=0.5, wireframe=True)
            scene = Scene(world)
            vis.render(scene)
            
            np.random.seed(3)
            
            f = 0
            widgets = [progressbar.Percentage(), progressbar.Bar()]
            bar = progressbar.ProgressBar(widgets=widgets, max_value=max_rays).start()
            history_args = {
                "bauble_radius": LSCdimX*0.005,
                "world_segment": "short",
                "short_length": LSCdimZ * 0.1,
                }
            k = 0
            if(convPlot):
                fig = plt.figure(num = 4, clear = True)
            xdata = []
            self.ydata = []
            ydataav = 0
            ydataavarr = []
            conv = 1
            self.convarr = []
            edge_emit = 0
            while k < max_rays:
            # while k < 1:
                for ray in scene.emit(1):
                # for ray in scene.emit(int(max_rays)):
                    steps = photon_tracer.follow(scene, ray, emit_method='redshift' )
                    path,surfnorms,events = zip(*steps)
                    if(len(path)<=2):
                        continue
                    if(self.enclosingBox.isChecked() and events[0]==Event.GENERATE and events[1]==Event.TRANSMIT and events[2] == Event.TRANSMIT and events[3] == Event.EXIT):
                        continue

                    entrance_rays.append(path[0])

                    # Draw every ray path in the web viewer.
                    vis.add_history(
                        steps,
                        baubles=False,
                        world_segment="short",
                        short_length=LSCdimZ * 0.1,
                        bauble_radius=LSCdimX * 0.005,
                        mark_final_position=True,
                        final_position_radius=0.03,
                    )
                    
                    if events[-1] == photon_tracer.Event.ABSORB:
                        exit_norms.append(surfnorms[-1])
                        exit_rays.append(path[-1])
                        absorbed_rays.append(path[-1])


                    elif events[-1] == photon_tracer.Event.KILL:
                        exit_norms.append(surfnorms[-1])
                        exit_rays.append(path[-1])
  
                    elif events[-1] == photon_tracer.Event.EXIT:
                        exit_norms.append(surfnorms[-2])
                        j = surfnorms[-2]
                        # Fix the condition order and add proper None checking
                        if j is not None and len(j) >= 3 and j[2] is not None and abs(j[2]) <= 0.5:
                            edge_emit+=1
                        exit_rays.append(path[-2]) 
                    f += 1
                    bar.update(f)
                    k+=1
                    xdata.append(k)
                    self.ydata.append(edge_emit/k)
                    ydataav = ydataav*.95 + edge_emit/k * .05
                    ydataavarr.append(ydataav)
                    conv = conv*.95 + abs(edge_emit/k - ydataav)*.05
                    self.convarr.append(conv)
                    if(convPlot):
                        fig = plt.figure(num = 4)
                        if(len(xdata)>2):
                            del xdata[0]
                            del self.ydata[0]
                            del ydataavarr[0]
                            del self.convarr[0]
                        plot(xdata, self.ydata, c='b')
                        plot(xdata, ydataavarr, c='r')
                        plt.grid(True)
                        plt.xlabel('num rays')
                        plt.ylabel('opt. eff')
                        plt.title('optical efficiency vs. rays generated')
                        plt.pause(0.00001)
                        
                        fig = plt.figure(num = 5)
                        plot(xdata, self.convarr, c = 'k')
                        plt.yscale('log')
                        plt.title('convergence')
                        plt.pause(0.00001)
                if(conv < convThres):
                    # numRays = k
                    break
            time.sleep(1)
            vis.render(scene)
            
            return entrance_rays, exit_rays, exit_norms, absorbed_rays, k
        
        def doRayTracingBatchParallel(numRays, convThres, showSim):
            """Batch-based parallel processing that actually works"""
            
            numRays = int(numRays)
            
            # Process in batches sequentially, but with parallelized analysis
            batch_size = max(100, numRays // 16)
            num_batches = (numRays + batch_size - 1) // batch_size
            
            print(f"Processing {numRays} rays in {num_batches} batches of ~{batch_size} rays each")
            
            all_entrance_rays = []
            all_exit_rays = []
            all_exit_norms = []
            all_absorbed_rays = []
            
            total_processed = 0
            edge_emit = 0
            conv = 1
            
            # Create visualization once
            vis = MeshcatRenderer(open_browser=showSim, transparency=False, opacity=0.5, wireframe=True)
            scene = Scene(world)
            vis.render(scene)
            
            # Process each batch sequentially (but faster than single rays)
            for batch_num in range(num_batches):
                current_batch_size = min(batch_size, numRays - total_processed)
                
                print(f"Processing batch {batch_num + 1}/{num_batches} ({current_batch_size} rays)")
                
                # Process this batch
                batch_entrance, batch_exit, batch_exit_norms, batch_absorbed, processed = processBatch(
                    scene, vis, current_batch_size, total_processed
                )
                
                # Combine results
                all_entrance_rays.extend(batch_entrance)
                all_exit_rays.extend(batch_exit)
                all_exit_norms.extend(batch_exit_norms)
                all_absorbed_rays.extend(batch_absorbed)
                
                total_processed += processed
                
                # Check convergence every batch
                batch_edge_emit = 0
                for norm in batch_exit_norms:
                    if norm is not None and len(norm) >= 3 and norm[2] is not None and abs(norm[2]) <= 0.5:
                        batch_edge_emit += 1
                
                edge_emit += batch_edge_emit
                current_efficiency = edge_emit / total_processed if total_processed > 0 else 0
                
                print(f"Batch {batch_num + 1} complete: {processed} rays, efficiency: {current_efficiency:.4f}")
                
                # Early convergence check (optional)
                if total_processed > 500:  # Only check after reasonable sample size
                    recent_efficiency = batch_edge_emit / processed if processed > 0 else 0
                    conv = conv * 0.95 + abs(current_efficiency - recent_efficiency) * 0.05
                    if conv < convThres:
                        print(f"Converged early at {total_processed} rays")
                        break
            
            # Final visualization update
            time.sleep(1)
            vis.render(scene)
            
            return all_entrance_rays, all_exit_rays, all_exit_norms, all_absorbed_rays, total_processed

        def processBatch(scene, vis, batch_size, seed_offset):
            """Process a batch of rays - this runs sequentially but faster than one-by-one"""
            
            np.random.seed(3 + seed_offset)  # Different seed for each batch
            
            batch_entrance = []
            batch_exit = []
            batch_exit_norms = []
            batch_absorbed = []
            
            history_args = {
                "bauble_radius": LSCdimX*0.005,
                "world_segment": "short",
                "short_length": LSCdimZ * 0.1,
            }
            
            processed = 0
            attempts = 0
            max_attempts = batch_size * 3  # Prevent infinite loops
            
            while processed < batch_size and attempts < max_attempts:
                attempts += 1
                
                for ray in scene.emit(1):
                    steps = photon_tracer.follow(scene, ray, emit_method='redshift')
                    path, surfnorms, events = zip(*steps)
                    
                    # Apply your existing filtering
                    if len(path) <= 2:
                        continue
                    if (self.enclosingBox.isChecked() and events[0] == Event.GENERATE and 
                        events[1] == Event.TRANSMIT and events[2] == Event.TRANSMIT and 
                        events[3] == Event.EXIT):
                        continue
                    
                    batch_entrance.append(path[0])
                    
                    # Draw every ray path in the web viewer.
                    vis.add_history(
                        steps,
                        baubles=False,
                        world_segment="short",
                        short_length=LSCdimZ * 0.1,
                        bauble_radius=LSCdimX * 0.005,
                    )
                    
# ...existing code...
                    if events[-1] == photon_tracer.Event.ABSORB:
                        batch_exit_norms.append(surfnorms[-1])
                        batch_exit.append(path[-1])

                        absorber_name = getattr(photon_tracer, "_last_absorber_name", "Unknown")
                        batch_absorbed.append((path[-1], absorber_name))
# ...existing code...
                    elif events[-1] == photon_tracer.Event.KILL:
                        batch_exit_norms.append(surfnorms[-1])
                        batch_exit.append(path[-1])
                    elif events[-1] == photon_tracer.Event.EXIT:
                        batch_exit_norms.append(surfnorms[-2])
                        batch_exit.append(path[-2])
                    
                    processed += 1
                    if processed >= batch_size:
                        break
            
            return batch_entrance, batch_exit, batch_exit_norms, batch_absorbed, processed

        def analyzeResults(entrance_rays, exit_rays, exit_norms, absorbed_rays):
            # Primary LSC (LSC1) results
            edge_emit = 0
            edge_emit_left = 0
            edge_emit_right = 0
            edge_emit_front = 0
            edge_emit_back = 0
            edge_emit_bottom = 0
            edge_emit_top = 0

            
            # Second LSC (LSC2/Waveguide) results
            edge_emit_lsc2 = 0
            edge_emit_left_lsc2 = 0
            edge_emit_right_lsc2 = 0
            edge_emit_front_lsc2 = 0
            edge_emit_back_lsc2 = 0
            edge_emit_bottom_lsc2 = 0
            edge_emit_top_lsc2 = 0
            
            entrance_wavs = []
            exit_wavs = []
            emit_wavs = []
            
            # Get LSC dimensions and positions for boundary checking
            LSCdimX = float(self.dimx.text())
            LSCdimY = float(self.dimy.text())
            LSCdimZ = float(self.dimz.text())
            
            enableSecondLSC = self.enableSecondLSC.isChecked()
            if enableSecondLSC:
                LSC2dimX = float(self.dimx2.text())
                LSC2dimY = float(self.dimy2.text())
                LSC2dimZ = float(self.dimz2.text())
                offsetX = float(self.lsc2_offsetX.text())
                offsetY = float(self.lsc2_offsetY.text())
                offsetZ = float(self.lsc2_offsetZ.text())

            def isPositionInLSC1(position):
                """Check if position is within LSC1 boundaries"""
                return (abs(position[0]) <= LSCdimX/2 and 
                        abs(position[1]) <= LSCdimY/2 and 
                        abs(position[2]) <= LSCdimZ/2)
            
            def isPositionInLSC2(position):
                """Check if position is within LSC2 boundaries"""
                if not enableSecondLSC:
                    return False
                return (abs(position[0] - offsetX) <= LSC2dimX/2 and 
                        abs(position[1] - offsetY) <= LSC2dimY/2 and 
                        abs(position[2] - offsetZ) <= LSC2dimZ/2)

            # Analyze each exit ray
            for index, k in enumerate(exit_norms):
                if k[2] != None:
                    exit_position = exit_rays[index].position
                    
                    # Determine which LSC this exit belongs to
                    is_from_lsc1 = isPositionInLSC1(exit_position)
                    is_from_lsc2 = isPositionInLSC2(exit_position)
                    
                    # Handle rotation cases for LSC1
                    if is_from_lsc1:
                        if((self.rotateY or self.rotateX) is False or self.enclosingBox.isChecked()):
                            if abs(k[2]) <= 0.5:
                                edge_emit += 1
                            if abs(k[0] - (-1)) < 0.1:
                                edge_emit_left += 1
                            if abs(k[0] - 1) < 0.1:
                                edge_emit_right += 1
                            if abs(k[1] - (-1)) < 0.1:
                                edge_emit_front += 1
                            if abs(k[1] - 1) < 0.1:
                                edge_emit_back += 1
                            if abs(k[2] + 1) < 0.1:
                                edge_emit_bottom += 1
                            if abs(k[2] - 1) < 0.1:
                                edge_emit_top += 1
                        elif self.rotateX is True:
                            if abs(k[1]) <= 0.5:
                                edge_emit += 1
                            if abs(k[0] - (-1)) < 0.1:
                                edge_emit_left += 1
                            if abs(k[0] - 1) < 0.1:
                                edge_emit_right += 1
                            if abs(k[2] - (-1)) < 0.1:
                                edge_emit_front += 1
                            if abs(k[2] - 1) < 0.1:
                                edge_emit_back += 1
                            if abs(k[1] + 1) < 0.1:
                                edge_emit_bottom += 1
                            if abs(k[1] - 1) < 0.1:
                                edge_emit_top += 1
                        elif self.rotateY is True:
                            if abs(k[0]) <= 0.5:
                                edge_emit += 1
                            if abs(k[2] - (-1)) < 0.1:
                                edge_emit_left += 1
                            if abs(k[2] - 1) < 0.1:
                                edge_emit_right += 1
                            if abs(k[1] - (-1)) < 0.1:
                                edge_emit_front += 1
                            if abs(k[1] - 1) < 0.1:
                                edge_emit_back += 1
                            if abs(k[0] + 1) < 0.1:
                                edge_emit_bottom += 1
                    
                    # Handle LSC2 analysis (assuming no rotation for simplicity)
                    elif is_from_lsc2:
                        if abs(k[2]) <= 0.5:
                            edge_emit_lsc2 += 1
                        if abs(k[0] - (-1)) < 0.1:
                            edge_emit_left_lsc2 += 1
                        if abs(k[0] - 1) < 0.1:
                            edge_emit_right_lsc2 += 1
                        if abs(k[1] - (-1)) < 0.1:
                            edge_emit_front_lsc2 += 1
                        if abs(k[1] - 1) < 0.1:
                            edge_emit_back_lsc2 += 1
                        if abs(k[2] + 1) < 0.1:
                            edge_emit_bottom_lsc2 += 1
                        if abs(k[2] - 1) < 0.1:
                            edge_emit_top_lsc2 += 1

            # Calculate total rays
            numRays = len(entrance_rays)
            total_exit_rays = len(exit_rays)

            # # Add absorbed ray positions and wavelengths
            # xpos_abs = []
            # ypos_abs = []
            # zpos_zbs = []
            # direction_abs = []
            # absorbed_wavs = []
            # for ray in absorbed_rays:
            #     absorbed_wavs.append(ray.wavelength)
            #     xpos_abs.append(ray.position[0])
            #     ypos_abs.append(ray.position[1])
            #     zpos_zbs.append(ray.position[2])
            #     direction_abs.append(ray.direction)

# ...existing code...
            # Keep only rays absorbed in the main LSC
            absorbed_rays = [r for r, absorber in absorbed_rays if absorber == "LSC"]

            absorbed_wavs = [r.wavelength for r in absorbed_rays]
            xpos_abs = [r.position[0] for r in absorbed_rays]
            ypos_abs = [r.position[1] for r in absorbed_rays]
            zpos_zbs = [r.position[2] for r in absorbed_rays]
            direction_abs = [r.direction for r in absorbed_rays]
# ...existing code...

            # Calculate actual absorption (only ABSORB events)
            actual_absorbed_rays = len(absorbed_rays)
            absorption_percentage = (actual_absorbed_rays / numRays) * 100 if numRays > 0 else 0


            # Save results to file
            if self.saveFileName != '':
                dataFile.write("\n=== PRIMARY LSC (LSC1) RESULTS ===\n")
                dataFile.write("Opt eff\t" + str(edge_emit/numRays) + "\n")
                dataFile.write("\t\tLeft \tRight \tFront \tBack\n")
                dataFile.write("Edge emission\t" + str(edge_emit_left/numRays) + " \t" + str(edge_emit_right/numRays) + " \t" + str(edge_emit_front/numRays) + " \t" + str(edge_emit_back/numRays) + "\n")
                dataFile.write("Bottom emission\t" + str(edge_emit_bottom/numRays) + "\t Absorption coeff " + str(-np.log10(max(edge_emit_bottom/numRays, 1e-10))/float(self.dimz.text())) + "\n")
                dataFile.write("Top emission\t" + str(edge_emit_top/numRays) + "\n")
                
                if enableSecondLSC:
                    dataFile.write("\n=== WAVEGUIDE LSC (LSC2) RESULTS ===\n")
                    dataFile.write("LSC2 Opt eff\t" + str(edge_emit_lsc2/numRays) + "\n")
                    dataFile.write("LSC2 Edge emission\t" + str(edge_emit_left_lsc2/numRays) + " \t" + str(edge_emit_right_lsc2/numRays) + " \t" + str(edge_emit_front_lsc2/numRays) + " \t" + str(edge_emit_back_lsc2/numRays) + "\n")
                    dataFile.write("Combined Opt eff\t" + str((edge_emit + edge_emit_lsc2)/numRays) + "\n")
                
                # Write ray data
                dataFile.write("type\tposx\tposy\tposz\tdirx\tdiry\tdirz\tsurfx\tsurfy\tsurfz\twav\tLSC\n")
                for ray in entrance_rays:
                    dataFile.write("entrance\t")
                    for k in range(3):
                        dataFile.write(str(ray.position[k]) + "\t")
                    for k in range(3):
                        dataFile.write(str(ray.direction[k]) + "\t")
                    for k in range(3):
                        dataFile.write('None \t')
                    dataFile.write(str(ray.wavelength) + "\tN/A\n")
                
                for index, ray in enumerate(exit_rays):
                    exit_position = ray.position
                    lsc_source = "LSC1" if isPositionInLSC1(exit_position) else ("LSC2" if isPositionInLSC2(exit_position) else "Unknown")
                    
                    dataFile.write("exit \t")
                    for k in range(3):
                        dataFile.write(str(ray.position[k]) + "\t")
                    for k in range(3):
                        dataFile.write(str(ray.direction[k]) + "\t")
                    for k in range(3):
                        dataFile.write(str(exit_norms[index][k]) + "\t")
                    dataFile.write(str(ray.wavelength) + "\t" + lsc_source + "\n")


            # Simplified detector analysis
            if enableSecondLSC and self.enableDetector.isChecked() and hasattr(self, 'current_detector') and self.current_detector:
                detector_hits = self.current_detector.detector_delegate.detected_count - self.detector_initial_count
                detector_efficiency = detector_hits / numRays * 100
                
                print(f"\n=== DETECTOR RESULTS ===")
                print(f"Detector hits: {detector_hits}")
                print(f"Detector efficiency: {detector_efficiency:.2f}%")
                
                if dataFile:
                    dataFile.write(f"\n=== DETECTOR RESULTS ===\n")
                    dataFile.write(f"Detector hits\t{detector_hits}\n")
                    dataFile.write(f"Detector efficiency\t{detector_efficiency:.2f}%\n")

            # Continue with existing wavelength and plotting analysis...
            # [Rest of the function remains the same for plotting]
            xpos_ent = []
            ypos_ent = []
            xpos_exit = []
            ypos_exit = []
            for ray in entrance_rays:
                entrance_wavs.append(ray.wavelength)
                xpos_ent.append(ray.position[0])
                ypos_ent.append(ray.position[1])
            for ray in exit_rays:
                exit_wavs.append(ray.wavelength)
                xpos_exit.append(ray.position[0])
                ypos_exit.append(ray.position[1])
            for k in range(len(exit_wavs)):
                if(exit_wavs[k]!=entrance_wavs[k]):
                    emit_wavs.append(exit_wavs[k])

            # Figure 10: Azimuthal angle histogram for absorbed rays
            plt.figure(10, clear=True)
            if direction_abs:  # Check if there are absorbed ray directions
                # Calculate azimuthal angles for absorbed rays
                absorbed_azimuthal_angles = []
                for direction in direction_abs:
                    # Azimuthal angle: atan2(y, x) in degrees
                    azimuthal_angle = np.degrees(np.arctan2(direction[1], direction[0]))
                    # Convert to 0-360 range
                    if azimuthal_angle < 0:
                        azimuthal_angle += 360
                    absorbed_azimuthal_angles.append(azimuthal_angle)
                
                plt.hist(absorbed_azimuthal_angles, bins=36, range=(0, 360), alpha=0.7, color='purple', edgecolor='black')
                plt.title(f'Absorbed rays azimuthal angle distribution ({len(absorbed_rays)} rays)')
                plt.xlabel('Azimuthal angle (degrees)')
                plt.ylabel('Number of absorbed rays')
                plt.grid(True, alpha=0.3)
                plt.xlim(0, 360)
                
                # Add statistics text
                az_mean = np.mean(absorbed_azimuthal_angles)
                az_std = np.std(absorbed_azimuthal_angles)
                plt.text(0.02, 0.98, 
                        f'Mean: {az_mean:.1f}°\nStd: {az_std:.1f}°\nTotal: {len(absorbed_azimuthal_angles)} rays', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                plt.text(0.5, 0.5, 'No rays absorbed', transform=plt.gca().transAxes, 
                        ha='center', va='center', fontsize=14)
                plt.title('Absorbed rays azimuthal angle distribution (No absorption)')
                plt.xlabel('Azimuthal angle (degrees)')
                plt.ylabel('Number of absorbed rays')
                plt.xlim(0, 360)

            if(self.saveFolder!=''):
                plt.savefig(self.saveFolder+"/"+"absorbed_azimuthal_histogram.png", dpi=figDPI)
            plt.pause(0.00001)

            # Figure 11: Polar angle histogram for absorbed rays
            plt.figure(11, clear=True)
            if direction_abs:  # Check if there are absorbed ray directions
                # Calculate polar angles for absorbed rays
                absorbed_polar_angles = []
                for direction in direction_abs:
                    # Polar angle: arccos(|z|) for 0-90 degree range
                    cos_theta = abs(direction[2])  # Use absolute value for 0-90° range
                    cos_theta = np.clip(cos_theta, 0, 1)  # Clamp to valid range for arccos
                    theta_rad = np.arccos(cos_theta)
                    theta_deg = np.degrees(theta_rad)
                    absorbed_polar_angles.append(theta_deg)
                
                plt.hist(absorbed_polar_angles, bins=90, range=(0, 90), alpha=0.7, color='orange', edgecolor='black')
                plt.title(f'Absorbed rays polar angle distribution ({len(absorbed_rays)} rays)')
                plt.xlabel('Polar angle (degrees)')
                plt.ylabel('Number of absorbed rays')
                plt.grid(True, alpha=0.3)
                plt.xlim(0, 90)
                
                # Add statistics text
                pol_mean = np.mean(absorbed_polar_angles)
                pol_std = np.std(absorbed_polar_angles)
                plt.text(0.02, 0.98, 
                        f'Mean: {pol_mean:.1f}°\nStd: {pol_std:.1f}°\nTotal: {len(absorbed_polar_angles)} rays', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                plt.text(0.5, 0.5, 'No rays absorbed', transform=plt.gca().transAxes, 
                        ha='center', va='center', fontsize=14)
                plt.title('Absorbed rays polar angle distribution (No absorption)')
                plt.xlabel('Polar angle (degrees)')
                plt.ylabel('Number of absorbed rays')
                plt.xlim(0, 90)

            if(self.saveFolder!=''):
                plt.savefig(self.saveFolder+"/"+"absorbed_polar_histogram.png", dpi=figDPI)
            plt.pause(0.00001)
                    
            plt.figure(7, clear=True)
            if zpos_zbs:  # Check if there are absorbed rays
                plt.hist(zpos_zbs, bins=50, range=(0, 17), alpha=0.7, color='blue', edgecolor='black')
                plt.title(f'Absorbed rays distribution along Z-axis ({len(absorbed_rays)} rays)')
                plt.xlabel('Z position (mm)')
                plt.ylabel('Number of absorbed rays')
                plt.grid(True, alpha=0.3)
                plt.xlim(0, 17)
                
                # Add statistics text
                z_mean = np.mean(zpos_zbs)
                z_std = np.std(zpos_zbs)
                plt.text(0.02, 0.98, 
                        f'Mean Z: {z_mean:.2f} mm\nStd Z: {z_std:.2f} mm\nTotal: {len(zpos_zbs)} rays', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                plt.text(0.5, 0.5, 'No rays absorbed', transform=plt.gca().transAxes, 
                        ha='center', va='center', fontsize=14)
                plt.title('Absorbed rays distribution along Z-axis (No absorption)')
                plt.xlabel('Z position (mm)')
                plt.ylabel('Number of absorbed rays')
                plt.xlim(0, 17)

            if(self.saveFolder!=''):
                plt.savefig(self.saveFolder+"/"+"absorption_z_histogram.png", dpi=figDPI)
            plt.pause(0.00001)

            # Histogram of absorbed rays along Y direction (-0.6 to 0.6)
            plt.figure(8, clear=True)
            if ypos_abs:  # Check if there are absorbed rays
                plt.hist(ypos_abs, bins=50, range=(-3, 3), alpha=0.7, color='green', edgecolor='black')
                plt.title(f'Absorbed rays distribution along Y-axis ({len(absorbed_rays)} rays)')
                plt.xlabel('Y position (mm)')
                plt.ylabel('Number of absorbed rays')
                plt.grid(True, alpha=0.3)
                
                # Add statistics text
                y_mean = np.mean(ypos_abs)
                y_std = np.std(ypos_abs)
                plt.text(0.02, 0.98, 
                        f'Mean Y: {y_mean:.2f} mm\nStd Y: {y_std:.2f} mm\nTotal: {len(ypos_abs)} rays', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                plt.text(0.5, 0.5, 'No rays absorbed', transform=plt.gca().transAxes, 
                        ha='center', va='center', fontsize=14)
                plt.title('Absorbed rays distribution along Y-axis (No absorption)')
                plt.xlabel('Y position (mm)')
                plt.ylabel('Number of absorbed rays')
                plt.xlim(-0.6, 0.6)

            if(self.saveFolder!=''):
                plt.savefig(self.saveFolder+"/"+"absorption_y_histogram.png", dpi=figDPI)
            plt.pause(0.00001)

            plt.figure(9, clear=True)
            if xpos_abs:  # Check if there are absorbed rays
                plt.hist(xpos_abs, bins=50, range=(-3, 3), alpha=0.7, color='red', edgecolor='black')
                plt.title(f'Absorbed rays distribution along X-axis ({len(absorbed_rays)} rays)')
                plt.xlabel('X position (mm)')
                plt.ylabel('Number of absorbed rays')
                plt.grid(True, alpha=0.3)
                plt.xlim(-3, 3)
                
                # Add statistics text
                x_mean = np.mean(xpos_abs)
                x_std = np.std(xpos_abs)
                plt.text(0.02, 0.98, 
                        f'Mean X: {x_mean:.2f} mm\nStd X: {x_std:.2f} mm\nTotal: {len(xpos_abs)} rays', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                plt.text(0.5, 0.5, 'No rays absorbed', transform=plt.gca().transAxes, 
                        ha='center', va='center', fontsize=14)
                plt.title('Absorbed rays distribution along X-axis (No absorption)')
                plt.xlabel('X position (mm)')
                plt.ylabel('Number of absorbed rays')
                plt.xlim(-2.5, 2.5)

            if(self.saveFolder!=''):
                plt.savefig(self.saveFolder+"/"+"absorption_x_histogram.png", dpi=figDPI)
            plt.pause(0.00001)

            if(self.saveFolder!=''):
                plt.savefig(self.saveFolder+"/"+"absorption_x_histogram.png", dpi=figDPI)
            plt.pause(0.00001)

            # ========== ADD THIS SECTION: SAVE RAW DATA ==========
            if self.saveFolder != '':
                print("\n=== SAVING RAW DATA ===")
                
                # 1. Save absorbed ray positions as CSV
                if len(absorbed_rays) > 0:
                    absorbed_data = pd.DataFrame({
                        'x_position': xpos_abs,
                        'y_position': ypos_abs,
                        'z_position': zpos_zbs,
                        'wavelength': absorbed_wavs,
                        'direction_x': [d[0] for d in direction_abs],
                        'direction_y': [d[1] for d in direction_abs],
                        'direction_z': [d[2] for d in direction_abs]
                    })
                    absorbed_csv_path = f"{self.saveFolder}/absorbed_rays_raw_data.csv"
                    absorbed_data.to_csv(absorbed_csv_path, index=False)
                    print(f"Saved absorbed ray positions to: {absorbed_csv_path}")
                
                # 2. Save YZ heatmap histogram data
                if absorbed_wavs:
                    # Recreate histogram (you already have counts, y_edges, z_edges from line ~1661)
                    # But recalculate here for clarity
                    counts_yz, y_edges_yz, z_edges_yz = np.histogram2d(
                        np.array(ypos_abs), np.array(zpos_zbs), bins=50
                    )
                    
                    # Save bin counts as CSV
                    heatmap_df = pd.DataFrame(counts_yz.T)  # Transpose to match plot orientation
                    heatmap_csv_path = f"{self.saveFolder}/absorption_heatmap_yz_counts.csv"
                    heatmap_df.to_csv(heatmap_csv_path, index=False, header=False)
                    
                    # Save bin edges separately
                    edges_data = {
                        'y_bin_edges': y_edges_yz,
                        'z_bin_edges': z_edges_yz
                    }
                    edges_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in edges_data.items()]))
                    edges_csv_path = f"{self.saveFolder}/absorption_heatmap_yz_edges.csv"
                    edges_df.to_csv(edges_csv_path, index=False)
                    
                    print(f"Saved YZ heatmap bin counts to: {heatmap_csv_path}")
                    print(f"Saved YZ heatmap bin edges to: {edges_csv_path}")
                
                # 3. Save 1D histogram data (Z, Y, X directions)
                if len(zpos_zbs) > 0:
                    # Z-axis histogram
                    counts_z, bins_z = np.histogram(zpos_zbs, bins=50, range=(0, 17))
                    z_hist_df = pd.DataFrame({
                        'bin_center': (bins_z[:-1] + bins_z[1:]) / 2,
                        'count': counts_z
                    })
                    z_hist_df.to_csv(f"{self.saveFolder}/absorption_z_histogram_data.csv", index=False)
                    
                    # Y-axis histogram
                    counts_y, bins_y = np.histogram(ypos_abs, bins=50, range=(-3, 3))
                    y_hist_df = pd.DataFrame({
                        'bin_center': (bins_y[:-1] + bins_y[1:]) / 2,
                        'count': counts_y
                    })
                    y_hist_df.to_csv(f"{self.saveFolder}/absorption_y_histogram_data.csv", index=False)
                    
                    # X-axis histogram
                    counts_x, bins_x = np.histogram(xpos_abs, bins=50, range=(-2.5, 2.5))
                    x_hist_df = pd.DataFrame({
                        'bin_center': (bins_x[:-1] + bins_x[1:]) / 2,
                        'count': counts_x
                    })
                    x_hist_df.to_csv(f"{self.saveFolder}/absorption_x_histogram_data.csv", index=False)
                    
                    print("Saved 1D histogram data for X, Y, Z axes")
                
                # 4. Save angular distribution data
                if len(direction_abs) > 0:
                    # Azimuthal angles
                    azimuthal_angles = []
                    for direction in direction_abs:
                        azimuthal_angle = np.degrees(np.arctan2(direction[1], direction[0]))
                        if azimuthal_angle < 0:
                            azimuthal_angle += 360
                        azimuthal_angles.append(azimuthal_angle)
                    
                    counts_az, bins_az = np.histogram(azimuthal_angles, bins=36, range=(0, 360))
                    az_hist_df = pd.DataFrame({
                        'bin_center': (bins_az[:-1] + bins_az[1:]) / 2,
                        'count': counts_az
                    })
                    az_hist_df.to_csv(f"{self.saveFolder}/absorption_azimuthal_histogram_data.csv", index=False)
                    
                    # Polar angles
                    polar_angles = []
                    for direction in direction_abs:
                        cos_theta = np.clip(abs(direction[2]), 0, 1)
                        theta_deg = np.degrees(np.arccos(cos_theta))
                        polar_angles.append(theta_deg)
                    
                    counts_pol, bins_pol = np.histogram(polar_angles, bins=90, range=(0, 90))
                    pol_hist_df = pd.DataFrame({
                        'bin_center': (bins_pol[:-1] + bins_pol[1:]) / 2,
                        'count': counts_pol
                    })
                    pol_hist_df.to_csv(f"{self.saveFolder}/absorption_polar_histogram_data.csv", index=False)
                    
                    print("Saved angular distribution data")
                
                print("=== RAW DATA SAVE COMPLETE ===\n")

            plt.figure(1, clear=True)
            
            # Plot absorbed rays as heatmap on YZ plane
            if absorbed_wavs:  # Only plot if there are absorbed rays
                # Extract Y and Z positions for absorbed rays
                ypos_abs_array = np.array(ypos_abs)
                zpos_abs_array = np.array(zpos_zbs)
                
                # Create 2D histogram - THIS IS YOUR GRID
                counts, y_edges, z_edges = np.histogram2d(
                    ypos_abs_array, zpos_abs_array, 
                    bins=50  # Adjust for coarser/finer grid
                )
                
                # Calculate effective absorbed area (36.8% threshold)
                max_count = np.max(counts)
                threshold = max_count * 0.05  # 10% of maximum (adjustable criterion)
                
                # Find bins that exceed the threshold
                effective_mask = counts >= threshold
                effective_bins = np.sum(effective_mask)
                
                # Calculate bin area
                y_bin_width = (y_edges[1] - y_edges[0])
                z_bin_width = (z_edges[1] - z_edges[0])
                bin_area = y_bin_width * z_bin_width  # mm²
                
                # Calculate effective absorbed area
                effective_absorbed_area = effective_bins * bin_area
                
                # Calculate total area covered by any absorption
                total_mask = counts > 0
                total_bins = np.sum(total_mask)
                total_absorbed_area = total_bins * bin_area
                
                # Calculate area efficiency (effective/total)
                area_efficiency = (effective_absorbed_area / total_absorbed_area * 100) if total_absorbed_area > 0 else 0
                
                # PRINT RESULTS
                print(f"\n=== ABSORPTION AREA ANALYSIS ===")
                print(f"Maximum absorption density: {max_count} rays/bin")
                print(f"10% threshold: {threshold:.1f} rays/bin")
                print(f"Effective absorbed area: {effective_absorbed_area:.3f} mm²")
                print(f"Total absorbed area: {total_absorbed_area:.3f} mm²")
                print(f"Area efficiency: {area_efficiency:.1f}%")
                print(f"Bin size: {y_bin_width:.4f} × {z_bin_width:.4f} mm")
                print(f"Grid resolution: {len(y_edges)-1} × {len(z_edges)-1} bins")
                
                # SIMPLE PIXELATED HEATMAP - NO INTERPOLATION
                # Use imshow with extent to map bins to axes correctly
                plt.imshow(
                    counts.T,  # Transpose to match Y (horizontal) and Z (vertical)
                    origin='lower',  # Put (0,0) at bottom-left
                    extent=[y_edges[0], y_edges[-1], z_edges[0], z_edges[-1]],  # Map bins to real coordinates
                    cmap='viridis',  # Color map
                    aspect='auto',  # Adjust aspect ratio to fill plot
                    interpolation='nearest'  # NO interpolation - sharp pixels
                )
                
                # Add colorbar
                cbar = plt.colorbar(label='Number of absorbed rays')
                
                # Optional: Add contour line for 10% threshold
                if threshold > 0:
                    # Create meshgrid for contour (centers of bins)
                    Y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                    Z_centers = (z_edges[:-1] + z_edges[1:]) / 2
                    Y, Z = np.meshgrid(Y_centers, Z_centers)
                    
                    threshold_contour = plt.contour(
                        Y, Z, counts.T, 
                        levels=[threshold], 
                        colors='red', 
                        linewidths=2,
                        linestyles='dashed'
                    )
                    plt.clabel(threshold_contour, inline=True, fontsize=10, fmt='10%%', colors='red')
                
                plt.title(f'Light absorption density on YZ plane ({len(absorbed_rays)} rays absorbed)')
                plt.xlabel('Y position (mm)')
                plt.ylabel('Z position (mm)')
                plt.grid(False)  # Turn off grid for cleaner look
                
                # Enhanced statistics text box with area analysis
                absorption_percentage = (len(absorbed_rays) / numRays) * 100
                plt.text(0.02, 0.98, 
                        f'Total rays: {numRays}\nAbsorbed: {len(absorbed_rays)}\nAbsorption: {absorption_percentage:.1f}%\n'
                        f'Effective area: {effective_absorbed_area:.3f} mm²\nTotal area: {total_absorbed_area:.3f} mm²\n'
                        f'Area efficiency: {area_efficiency:.1f}%', 
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Save area analysis to file if dataFile exists
                if dataFile:
                    dataFile.write(f"\n=== ABSORPTION AREA ANALYSIS ===\n")
                    dataFile.write(f"Maximum absorption density\t{max_count} rays/bin\n")
                    dataFile.write(f"10% threshold\t{threshold:.1f} rays/bin\n")
                    dataFile.write(f"Effective absorbed area\t{effective_absorbed_area:.3f} mm²\n")
                    dataFile.write(f"Total absorbed area\t{total_absorbed_area:.3f} mm²\n")
                    dataFile.write(f"Area efficiency\t{area_efficiency:.1f}%\n")
                    dataFile.write(f"Bin size\t{y_bin_width:.4f} × {z_bin_width:.4f} mm\n")
                    dataFile.write(f"Grid resolution\t{len(y_edges)-1} × {len(z_edges)-1} bins\n")
                    
            else:
                plt.text(0.5, 0.5, 'No rays absorbed', transform=plt.gca().transAxes, 
                        ha='center', va='center', fontsize=14)
                plt.title('Light absorption density on YZ plane (No absorption detected)')
                plt.xlabel('Y position (mm)')
                plt.ylabel('Z position (mm)')
                
                print("\n=== ABSORPTION AREA ANALYSIS ===")
                print("No absorption detected - cannot calculate effective area")
            
            if(self.saveFolder!=''):
                plt.savefig(self.saveFolder+"/"+"absorption_heatmap_yz.png", dpi=figDPI, bbox_inches='tight')
            plt.pause(0.00001)
            


            
            # plt.figure(2, clear = True)
            # n, bins, patches = hist(entrance_wavs, bins = 10, histtype = 'step', label='entrance wavs')
            # plot(wavelengths, intensity/max(intensity)*max(n))
            # plt.title('Entrance wavelengths')
            # plt.legend()
            # if(self.saveFolder!=''):
            #     plt.savefig(self.saveFolder+"/"+"entrance_wavs.png", dpi=figDPI)
            # plt.pause(0.00001)
                    
            # plt.figure(3, clear=True)
            # n, bins, patches = hist(emit_wavs, bins = 10, histtype = 'step', label='emit wavs')
            # if(self.lumophore.currentText() != 'None' ):
            #     plot(x, abs_spec*max(n), label = 'LR305 abs')
            #     plot(x, ems_spec*max(n), label = 'LR305 emis')
            # plt.title('Re-emitted light wavelengths')
            # plt.legend()
            # if(self.saveFolder!=''):
            #     plt.savefig(self.saveFolder+"/"+"emit_wavs.png", dpi=figDPI)
            # plt.pause(0.00001)
            
            # # if(convPlot):
            # plt.figure(4)
            # if(not convPlot):
            #     plot(range(len(entrance_rays)), self.ydata)
            # plt.title('optical efficiency vs. rays generated')
            # plt.grid(True)
            # plt.xlabel('num rays')
            # plt.ylabel('opt. eff')
            # if(self.saveFolder!=''):
            #     plt.savefig(self.saveFolder+"/"+"conv_plot.png", dpi=figDPI)
            # plt.pause(0.00001)
            
            # plt.figure(5)
            # if(not convPlot):
            #     plot(range(len(entrance_rays)), self.convarr)
            # plt.title('convergence')
            # plt.grid(True)
            # plt.xlabel('num rays')
            # plt.ylabel('convergence parameter')
            # plt.yscale('log')
            # if(self.saveFolder!=''):
            #     plt.savefig(self.saveFolder+"/"+"conv_plot2.png", dpi=figDPI)
            # plt.pause(0.00001)

            # fig = plt.figure(6, clear=True, figsize=(3, 10))
            # fig.add_subplot(515)
            # norm = plt.Normalize(*(wavMin,wavMax))
            # wl = np.arange(wavMin, wavMax+1,2)
            # colorlist = list(zip(norm(wl), [np.array(wavelength_to_rgb(w))/255 for w in wl]))
            # spectralmap = matplotlib.colors.LinearSegmentedColormap.from_list("spectrum", colorlist)
            # colors_ent = [spectralmap(norm(value)) for value in entrance_wavs]
            # colors_exit = [spectralmap(norm(value)) for value in exit_wavs]
            # scatter(xpos_ent, ypos_ent, alpha=1.0, color=colors_ent)
            # scatter(xpos_exit, ypos_exit, alpha=1.0, color=colors_exit)
            # # plt.title('entrance/exit positions')
            # plt.xlabel('x position')
            # plt.ylabel('y position')
            # plt.axis('equal')
            # plt.ylim(-2, 2)
            # # plt.title('Entrance/exit ray positions')
            # plt.tight_layout()

            # fig.add_subplot(513)
            # n, bins, patches = hist(entrance_wavs, bins = 10, histtype = 'step', label='entrance wavs')
            # plot(wavelengths, intensity/max(intensity)*max(n))
            # # plt.title('Entrance wavelengths')
            # plt.xlabel('wavelength (nm)')
            # plt.ylabel('counts (entrance)')
            # plt.legend()
            # plt.grid()
            # plt.pause(0.00001)
            # plt.tight_layout()
            
            # fig.add_subplot(514)
            # n, bins, patches = hist(emit_wavs, bins = 10, histtype = 'step', label='emit wavs')
            # if(self.lumophore.currentText() != 'None' ):
            #     plot(x, abs_spec*max(n), label = 'LR305 abs')
            #     plot(x, ems_spec*max(n), label = 'LR305 emis')
            # # plt.title('Re-emitted light wavelengths')
            # plt.xlabel('wavelength (nm)')
            # plt.ylabel('counts (re-emitted)')
            # plt.legend(loc='upper left', fontsize='small')
            # plt.grid()
            # plt.pause(0.00001)
            # plt.tight_layout()
            
            # fig.add_subplot(511)
            # if(not convPlot):
            #     plot(range(len(entrance_rays)), self.ydata)
            # # plt.title('optical efficiency vs. rays generated')
            # plt.grid(True)
            # plt.xlabel('num rays')
            # plt.ylabel('opt. eff.')
            # plt.pause(0.00001)
            # plt.tight_layout()
            
            # fig.add_subplot(512)
            # if(not convPlot):
            #     plot(range(len(entrance_rays)), self.convarr)
            # # plt.title('convergence')
            # plt.grid(True)
            # plt.xlabel('num rays')
            # plt.ylabel('convergence')
            # plt.yscale('log')
            # plt.pause(0.00001)
            # plt.tight_layout()

            # if(self.saveFolder!=''):
            #     plt.savefig(self.saveFolder+"/"+"plots.png", dpi=figDPI)

            

        #%% define inputs
        wavMin = float(self.wavMin.text())
        wavMax = float(self.wavMax.text())
        LSCdimX = float(self.dimx.text())
        LSCdimY = float(self.dimy.text())
        LSCdimZ = float(self.dimz.text())
        LSCshape = self.inputShape.currentText()
        thinFilm = self.thinFilm.isChecked()
        thinFilmThick = float(self.thinFilmThickness.text())
        LumType = self.lumophore.currentText()
        LumConc = float(self.lumophoreConc.text())
        LumPLQY = float(self.lumophorePLQY.text())
        wavAbs = float(self.waveguideAbs.text())
        wavN = float(self.waveguideN.text())
        lightWavMin = float(self.lightWavMin.text())
        lightWavMax = float(self.lightWavMax.text())
        lightPattern = self.lightPattern.currentText()
        lightDimX = float(self.lightDimx.text())
        lightDimY = float(self.lightDimy.text())
        lightDiv = float(self.lightDiv.text())
        numRays = float(self.numRays.text())
        figDPI = int(self.figDPI.text())
        solAll = self.solarFaceAll.isChecked()
        solLeft = self.solarFaceLeft.isChecked()
        solRight = self.solarFaceRight.isChecked()
        solFront = self.solarFaceFront.isChecked()
        solBack = self.solarFaceBack.isChecked()
        bottomMir = self.bottomMir.isChecked()
        bottomScat = self.bottomScat.isChecked()
        enclosingBox = self.enclosingBox.isChecked()
        LSCbounds = self.LSCbounds
        convPlot = self.convPlot.isChecked()
        convThres = float(self.convThres.text())
        showSim = self.showSim.isChecked()
        
        maxZ = LSCdimZ
        if(LSCshape=='Sphere'):
            maxZ = LSCdimX
        
        world = createWorld(max(LSCdimX, LSCdimY, maxZ))

        
        # Create world with larger dimensions to accommodate both LSCs
        if enableSecondLSC:
            maxDim = max(LSCdimX, LSCdimY, maxZ, LSC2dimX, LSC2dimY, LSC2dimZ)
            world = createWorld(maxDim * 2)
        else:
            world = createWorld(max(LSCdimX, LSCdimY, maxZ))
        
        # ADD PVTRACE FIXES - Insert after world creation around line 1301
        import pvtrace.algorithm.photon_tracer as photon_tracer
        from pvtrace.algorithm.photon_tracer import next_hit, Event, find_container

        original_next_hit = photon_tracer.next_hit
        original_follow = photon_tracer.follow
        original_find_container = photon_tracer.find_container

        def corrected_find_container(intersections):

            filtered = [
                x for x in intersections
                if "Detector" not in x.hit.name
            ]

            if len(filtered) == 0:
                return original_find_container(intersections)

            return original_find_container(filtered)
        
        # Apply the fix
        # photon_tracer.find_container = corrected_find_container


        def corrected_follow(scene, ray, maxsteps=1000, maxpathlength=np.inf, emit_method='kT'):
            count = 0
            history = [(ray, (None,None,None), Event.GENERATE)]
            
            
            while True:
                count += 1
                if count > maxsteps or ray.travelled > maxpathlength:
                    history.append((ray, (None,None,None), Event.KILL))
                    break
            
                info = original_next_hit(scene, ray)  # Use original PVTrace version, since the corrected_find_container is now in place, corrected_next_hit logic is already integrated into the find_container fix
                if info is None:
                    history.append((ray, (None,None,None), Event.EXIT))
                    break

                hit, (container, adjacent), point, full_distance = info


                # Check if hit object is a detector
                if hasattr(hit.geometry, '__class__') and hit.geometry.__class__.__name__ == 'PlanarDetector':
                    # Check if ray is approaching from detection direction
                    if hasattr(hit.geometry.material.surface, 'delegate') and hasattr(hit.geometry.material.surface.delegate, '_is_detection_direction'):
                        if hit.geometry.material.surface.delegate._is_detection_direction(ray.direction):
                            # Ray hits detector - record detection and break
                            ray = ray.propagate(full_distance)
                            hit.geometry.material.surface.delegate.detected_count += 1
                            hit.geometry.material.surface.delegate.detected_rays.append({
                                'position': ray.position,
                                'direction': ray.direction,
                                'wavelength': ray.wavelength
                            })
                            history.append((ray, (None,None,None), Event.DETECT))
                            break

                if hit is scene.root:
                    history.append((ray.propagate(full_distance), (None,None,None), Event.EXIT))
                    break

                # BRUTAL FIX: Force adjacent to be LSC when exiting any waveguide
                # Since waveguides are embedded in LSC and separated from each other,
                # any waveguide exit should interact with LSC material, not other waveguides

                corrected_adjacent = adjacent
                
                if "Waveguide" in container.name and container.name == hit.name:
                    # Ray is inside a waveguide hitting its own surface (exiting)
                    # Force adjacent to be the LSC absorber (parent)
                    for node in scene.root.children:
                        if node.name == "LSC":
                            corrected_adjacent = node
                            break
                
                # # FIX: Correct adjacent detection for waveguide surfaces in the air
                # if hit.name == "LSC2_Waveguide" and container.name == "LSC2_Waveguide":
                #     # Ray is inside waveguide hitting waveguide surface
                #     # Adjacent should always be World (air), not absorber
                #     corrected_adjacent = scene.root  # Now world exists in scope
                # else:
                #     # Use original adjacent for other cases
                #     corrected_adjacent = adjacent
                
                material = container.geometry.material

                # Pentacene host: absorb only along z
                if container.name == "LSC":
                    z_component_fraction = abs(ray.direction[2])
                    absorption_distance = (
                        full_distance * z_component_fraction
                    )

                    absorbed, at_z_distance = material.is_absorbed(
                        ray,
                        absorption_distance
                    )

                    if absorbed:
                        at_distance = (
                            at_z_distance / z_component_fraction
                            if z_component_fraction > 1e-12
                            else full_distance
                        )
                    else:
                        at_distance = full_distance

                # Ce:YAG waveguides: normal Beer-Lambert
                else:

                    absorbed, at_distance = material.is_absorbed(
                        ray,
                        full_distance
                    )

                if absorbed and at_distance < full_distance:
                    ray = ray.propagate(at_distance)
                    component = material.component(ray.wavelength)

                    if component is not None and component.is_radiative(ray):
                        ray = component.emit(ray.representation(scene.root, container), method=emit_method)
                        ray = ray.representation(container, scene.root)
                        if isinstance(component, Luminophore):
                            event = Event.EMIT
                        elif isinstance(component, Scatterer):
                            event = Event.SCATTER
                        else:
                            event = Event.SCATTER
                        history.append((ray, (None,None,None), event))
                        
                        # Reset accumulated distance after emission (new ray path)
                        accumulated_absorber_distance = 0.0
                        continue
                    else:
                        # Non-radiative absorption - ray is absorbed
                        setattr(photon_tracer, "_last_absorber_name", container.name)
                        history.append((ray, (None,None,None), Event.ABSORB))
                        break
                else:
                    ray = ray.propagate(full_distance)
                    surface = hit.geometry.material.surface
                    ray = ray.representation(scene.root, hit)
                    
                    # Use corrected adjacent for surface interactions
                    if surface.is_reflected(ray, hit.geometry, container, corrected_adjacent): #Not corrected_adjacent!!
                        ray = surface.reflect(ray, hit.geometry, container, corrected_adjacent)
                        ray = ray.representation(hit, scene.root)
                        
                        try:
                            local_pos = list(np.array(ray.position) - np.array(hit.location))
                            normal = hit.geometry.normal(local_pos)
                        except:
                            normal = (None, None, None)
                            
                        history.append((ray, normal, Event.REFLECT))
                        continue
                    else:
                        ref_ray = surface.transmit(ray, hit.geometry, container, corrected_adjacent)
                        if ref_ray is None:
                            history.append((ray, (None,None,None), Event.KILL))
                            break
                            
                        ray = ref_ray
                        ray = ray.representation(hit, scene.root)
                        
                        try:
                            local_pos = list(np.array(ray.position) - np.array(hit.location))
                            normal = hit.geometry.normal(local_pos)
                        except:
                            normal = (None, None, None)
                            
                        history.append((ray, normal, Event.TRANSMIT))
                        continue
                        
            return history

        # Apply the fixes
        # photon_tracer.find_container = corrected_find_container
        photon_tracer.follow = corrected_follow

        if(enclosingBox):
            enclBox = createBoxLSC(LSCdimX*1.32, LSCdimY*1.32, LSCdimZ*1.1,0,wavN)
            if(len(widget.LSCbounds)>0):
                
                enclBox.location = [ (self.LSCbounds[0][0] + LSCbounds[1][0])/2, (LSCbounds[0][1] + LSCbounds[1][1])/2, 0]
            enclBox.name = "enclBox"
            enclBox.geometry.material.refractive_index=1.0
            del enclBox.geometry.material.components[0:2]
            enclBox.geometry.material.surface = Surface(delegate = NullSurfaceDelegate())
        
        if not thinFilm:
            if(LSCshape == 'Box'):
                LSC = createBoxLSC(LSCdimX, LSCdimY, LSCdimZ, wavAbs, wavN)
            if(LSCshape == 'Cylinder'):
                LSC = createCylLSC(LSCdimX, LSCdimZ, wavAbs, wavN)
            if(LSCshape == 'Sphere'):
                LSC = createSphLSC(LSCdimX, wavAbs, wavN)
            if(LSCshape == 'Import Mesh'):
                LSC = createMeshLSC(self, wavAbs, wavN)
                # if(not np.isclose(LSC.location, LSC.geometry.trimesh.centroid).all()):
                #     LSC.translate(-LSC.geometry.trimesh.centroid)
                # LSCmeshdims = LSC.geometry.trimesh.extents
                # if(self.rotateY):
                #     LSC.rotate(np.radians(90),(0,1,0))
                # if(self.rotateX):
                #     LSC.rotate(np.radians(90),(1,0,0))
                # if(LSCmeshdims[0] < LSCmeshdims[2]):
                #     LSC.rotate(np.radians(90),(0,1,0))
                #     temp = LSCdimZ
                #     LSCdimZ = LSCdimX
                #     LSCdimX = temp
                #     lightDimX = LSCdimX
                #     maxZ = LSCdimZ
                # elif(LSCmeshdims[1] < LSCmeshdims[2]):
                #     LSC.rotate(np.radians(90),(1,0,0))
                #     temp = LSCdimZ
                #     LSCdimZ = LSCdimY
                #     LSCdimY = temp
                #     lightDimY = LSCdimY
                #     maxZ = LSCdimZ
            # ✅ NEW: Position LSC so its bottom is 0.1mm BELOW world center
            LSC.location = (0, 0, 0)
        else:
            if(LSCshape == 'Box'):
                LSC = createBoxLSC(LSCdimX, LSCdimY, thinFilmThick, wavAbs, wavN)
                bulk_undoped = createBoxLSC(LSCdimX, LSCdimY, LSCdimZ, wavAbs, wavN)
            if(LSCshape == 'Cylinder'):
                LSC = createCylLSC(LSCdimX, thinFilmThick, wavAbs, wavN)
                bulk_undoped = createCylLSC(LSCdimX, LSCdimZ, wavAbs, wavN)
            if(LSCshape == 'Import Mesh'):
                LSC = createMeshLSC(self, wavAbs, wavN)
                LSC.geometry.trimesh.apply_scale(1,1,thinFilmThick/LSCdimZ)
                bulk_undoped = createMeshLSC(self, wavAbs, wavN)
            LSC.location = (0,0,LSCdimZ/2)
            bulk_undoped.name = "bulk"
            
        if(LumType == 'Lumogen Red'):
            LSC, x, abs_spec, ems_spec = addLR305(LSC, LumConc, LumPLQY)
            
        
        LSC = addSolarCells(LSC, solLeft, solRight, solFront, solBack, solAll)
        
        LSC = addBottomSurf(LSC, bottomMir, bottomScat)

        # # Add 2 air spacers below the LSC to avoid mis interactions.
        # side_width = (LSCdimY - 0.6)/2
        # thickness = LSCdimZ/50
        # bottomSpacer1 = createBoxLSC(LSCdimX, (LSCdimY-0.6)/2, LSCdimZ/50, 0, 1)
        # bottomSpacer1.name = "bottomSpacer1"
        # bottomSpacer1.location = [0, 0.31+side_width/2, -thickness/2]

        # bottomSpacer2 = createBoxLSC(LSCdimX, (LSCdimY-0.6)/2, LSCdimZ/50, 0, 1)
        # bottomSpacer2.name = "bottomSpacer2"
        # bottomSpacer2.location = [0, -0.31-side_width/2, -thickness/2]
# # ...existing code...

#         # Add optional XY mirror
#         xy_mirror = addXYMirror(
#             mirror_z=-72.0,
#             mirror_x=LSCdimX * 2,
#             mirror_y=LSCdimY * 2
#         )
#         print(f"XY mirror created at z=-72.0 mm")
# # ...existing code...
        # Create second LSC (waveguide) with automatic multi-part splitting
        if enableSecondLSC:
            waveguide_parent = None
            waveguide_parts = []
            
            if(LSC2shape == 'Box'):
                LSC2 = createBoxLSC(LSC2dimX, LSC2dimY, LSC2dimZ, wavAbs2, wavN2)
                LSC2.location = [offsetX, offsetY, offsetZ]
                LSC2.name = "LSC2_Waveguide"
                waveguide_parts = [LSC2]
            elif(LSC2shape == 'Cylinder'):
                LSC2 = createCylLSC(LSC2dimX, LSC2dimZ, wavAbs2, wavN2)
                LSC2.location = [offsetX, offsetY, offsetZ]
                LSC2.name = "LSC2_Waveguide"
                waveguide_parts = [LSC2]
            elif(LSC2shape == 'Sphere'):
                LSC2 = createSphLSC(LSC2dimX, wavAbs2, wavN2)
                LSC2.location = [offsetX, offsetY, offsetZ]
                LSC2.name = "LSC2_Waveguide"
                waveguide_parts = [LSC2]
            elif(LSC2shape == 'Import Mesh'):
                # Use the new multi-part mesh function
                # This automatically splits disconnected parts into separate nodes
                waveguide_parent, waveguide_parts = createMultiPartMeshLSC(
                    self, wavAbs2, wavN2, self.STLfile2, world
                )
                
                # Position the parent container (all parts move together)
                waveguide_parent.location = [offsetX, offsetY, offsetZ]
                
                # For backward compatibility, set LSC2 to the parent
                LSC2 = waveguide_parent
                
                print(f"Created {len(waveguide_parts)} waveguide part(s)")
            
            # CREATE DETECTOR HERE - AFTER WORLD AND LSCs ARE CREATED
            if self.enableDetector.isChecked():
                # Get detector parameters
                detector_x = float(self.detector_posX.text())
                detector_y = float(self.detector_posY.text())
                detector_z = float(self.detector_posZ.text())
                
                # Use LSC2 dimensions for detector size
                detector_length = LSC2dimX
                detector_width = LSC2dimY
                
                # Get detection direction
                dir_text = self.detector_direction.currentText()
                if dir_text == "Down (-Z)":
                    normal = (0, 0, 1)
                    detection_direction = (0, 0, -1)
                elif dir_text == "Up (+Z)":
                    normal = (0, 0, -1)
                    detection_direction = (0, 0, 1)
                elif dir_text == "Left (-X)":
                    normal = (1, 0, 0)
                    detection_direction = (-1, 0, 0)
                elif dir_text == "Right (+X)":
                    normal = (-1, 0, 0)
                    detection_direction = (1, 0, 0)
                elif dir_text == "Front (-Y)":
                    normal = (0, 1, 0)
                    detection_direction = (0, -1, 0)
                elif dir_text == "Back (+Y)":
                    normal = (0, -1, 0)
                    detection_direction = (0, 1, 0)
                else:
                    # Default
                    normal = (0, 0, 1)
                    detection_direction = (0, 0, -1)
                
                # Create detector using the simple method from LED_intensity.py
                detector = create_planar_detector_node(
                    name="LSC2_Detector",
                    length=detector_length,
                    width=detector_width,
                    normal=normal,
                    detection_direction=detection_direction,
                    parent=world
                )
                
                # Position the detector
                detector.translate((detector_x, detector_y, detector_z))
                print(f"Detector created at position: ({detector_x}, {detector_y}, {detector_z})")
                print(f"Detector size: {detector_length} x {detector_width}")
                print(f"Detection direction: {dir_text}")
                
                # Store initial detector count and make accessible
                self.current_detector = detector
                self.detector_initial_count = detector.detector_delegate.detected_count
            else:
                self.current_detector = None
                self.detector_initial_count = 0
            
            # Add lumophore to waveguide parts if needed
            if(LumType2 == 'Lumogen Red'):
                # Apply lumophore to all waveguide parts
                for part in waveguide_parts:
                    part, x2, abs_spec2, ems_spec2 = addLR305(part, LumConc2, LumPLQY2)

            if(LumType2 == 'Import Custom'):
                # Apply lumophore to all waveguide parts
                for part in waveguide_parts:
                    part, x2, abs_spec2, ems_spec2 = addCeYAG(part, LumConc2, LumPLQY2)
            
            # Configure waveguide surfaces for all parts
            for part in waveguide_parts:
                part = addWaveguideSurfaces(part)
        
        wavelengths, intensity, light = initLight(lightWavMin, lightWavMax)
        if(lightPattern == 'Rectangle Mask'):
            light = addRectMask(light, lightDimX, lightDimY)
        if(lightPattern == 'Circle Mask'):
            light = addCircMask(light, lightDimX)
        if(lightPattern == 'Point Source'):
            light = addPointSource(light)
        if(0<lightDiv<=180):
            light = addLightDiv(light, lightDiv)
        if lightDiv == 0:
            light = addStraightRays(light)  # Straight rays for lightDiv = 0
        if lightDiv == 1234567:
            light = addRealLambertian(light) # Lambertian with Fresnel for special case
            
        start_t = time.time()

        # entrance_rays, exit_rays, exit_norms, absorbed_rays, numRays = doRayTracing(numRays, convThres, showSim)
        use_parallel = numRays > 500  # Use parallel for larger simulations
        entrance_rays, exit_rays, exit_norms, absorbed_rays, numRays = doRayTracing(numRays, convThres, showSim, use_parallel)
        # Print timing results
        print(f"Took {time.time() - start_t}s.")

        analyzeResults(entrance_rays, exit_rays, exit_norms, absorbed_rays)
        return entrance_rays, exit_rays, exit_norms
        
    def resetPVTraceState(self):
        """Reset PVTrace to original state between simulations"""
        print("Resetting PVTrace state for new simulation...")
        
        # Restore original PVTrace functions
        import pvtrace.algorithm.photon_tracer as photon_tracer
        
        # Re-import to get fresh original functions
        import importlib
        importlib.reload(photon_tracer)
        
        # Clear any cached detector states
        if hasattr(self, 'current_detector'):
            self.current_detector = None
        if hasattr(self, 'detector_initial_count'):
            self.detector_initial_count = 0
        
        # Clear any other persistent states
        if hasattr(self, 'scene'):
            delattr(self, 'scene')
        
        print("PVTrace state reset complete")

#%% main
if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QApplication.instance()
    if app == None:
        app = QApplication([])
    widget = testingQT()
    widget.show()
    app.exec_()
    
    
