#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:25:53 2023

@author: albertsmith
"""

import os
from copy import copy


class CellReader():
    def __init__(self,filename:str):
        self.f=open(filename,'r')
        self.reset()
    
    def reset(self):
        self.f.seek(0)
        self.header=[]
        self._footer=None
        for line in self.f:
            self.header.append(line)
            if '"cells"' in line:break
        self.cont=True
        self.last_cell=[]
    
    def ReadCell(self):
        """
        Reads the next cell and returns all lines in that cell

        Returns
        -------
        out : TYPE
            DESCRIPTION.

        """
        f=self.f
        
        if not(self.cont):return None #End of cells already reached
        
        for line in f:
            if ']' in line:
                self.cont=False #Cell block is closed
                return None
            if '{' in line:
                break #New Cell has started
        
        out=[]
        count=1
        for line in f:
            count+=line.count('{')
            count-=line.count('}')
            # if '{' in line:count+=1
            # if '}' in line:count-=1
            if count<1:break
            out.append(line)
        self.last_cell=out
        
        
        return out

    @property
    def footer(self):
        if self._footer is None:
            self._footer=['],\n']
            while self.cont:self.ReadCell()
            for line in self.f:
                self._footer.append(line)
        return self._footer
        
        
    def get_source(self,lines:list=None):
        """
        Extracts the source information from the previously extracted cell or
        the provided lines

        Parameters
        ----------
        lines : list, optional
            List of lines to extract source from. The default is None.

        Returns
        -------
        list

        """
        if lines is None:lines=self.last_cell
        lines=copy(lines)
        
        while len(lines):
            line=lines.pop(0)
            if '"source"' in line:break
        count=1
        out=[]
        while len(lines):
            line=lines.pop(0)
            if '[' in line:count+=1
            if ']' in line:count-=1
            if count==0:break
            out.append(line)
        
        return out
        
    def __next__(self):
        if self.cont:
            return self.ReadCell()
        else:
            raise StopIteration
    
    def __iter__(self):
        self.reset()
        return self
    
    def __exit__(self):
        self.f.close()
        

def write_cell(f,lines:list,colab=False):
    """
    Write lines to a cell for a file

    Parameters
    ----------
    f : file handle
        File to be written to
    lines : list
        Lines to go into the cell.

    Returns
    -------
    None.

    """

    f.write('{\n')
    for line in lines:
        if colab:
            line=line.replace('<font','<img src=\\"https://raw.githubusercontent.com/alsinmr/SLEEPY_tutorial/033b817f027ebdcd6493a1f42ab9fdec290dbee8/JupyterBook/favicon.png\\"  width=40> <font')
        f.write(line)
    f.write('}')


def write_colab_setup(f):
    """
    Append lines to code to set up Google colab with pyDR

    Parameters
    ----------
    f : file handle

    Returns
    -------
    None.

    """
    
    
    
    f.write("""{
     "cell_type": "code",
     "execution_count": 0,
     "metadata": {},
     "outputs": [],
     "source": [
      "# SETUP SLEEPY\\n",
      "!git clone https://github.com/alsinmr/SLEEPY.git"
     ]
    }""")
    
def write_book_setup(f):
    """
    Append lines to code to set up pyDR for r 

    Parameters
    ----------
    f : file handle

    Returns
    -------
    None.

    """
    
    
    
    # f.write("""{
    #  "cell_type": "code",
    #  "execution_count": 0,
    #  "id": "759eab0f",
    #  "metadata": {},
    #  "outputs": [],
    #  "source": [
    #   "# SETUP SLEEPY\\n",
    #   "import os\\n",
    #   "os.chdir('../..')"
    #  ]
    # }""")
    
    f.write("""{
     "cell_type": "code",
     "execution_count": 0,
     "metadata": {},
     "outputs": [],
     "source": [
      "# SETUP SLEEPY"
     ]
    }""")
    

def add_links(f,filename):
    """
    Adds links (Colab, download) to the notebook for the website

    Parameters
    ----------
    f : TYPE
        DESCRIPTION.
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    chapter=filename.split('Chapters/Chapter')[1].split('/')[0]
    filename=os.path.split(filename)[1]
    f.write("""{
     "cell_type": "markdown",
     "metadata": {},
     "source": [""")
    f.write(f"""
      "<a href=\\"https://githubtocolab.com/alsinmr/SLEEPY_tutorial/blob/main/ColabNotebooks/Chapter{chapter}/{filename}\\" target=\\"_blank\\"><img src=\\"https://colab.research.google.com/assets/colab-badge.svg\\"></a>"
     """)
    # f.write(f"""
    #         "\\n\\n<a href=\\"https://github.com/alsinmr/pyDR_tutorial/raw/main/{filename}\\" target=\\"_blank\\"><img src=\\"Download-button.png\\" width=\\"100\\"></a>"
    #         """)
    f.write("""
            ]\n}""")
            
def add_image(f):
    f.write("""{
     "cell_type": "markdown",
     "metadata": {},
     "source": [
     "<img src=\\"https://raw.githubusercontent.com/alsinmr/SLEEPY_tutorial/033b817f027ebdcd6493a1f42ab9fdec290dbee8/JupyterBook/favicon.png\\"  width=50>"
     ]
    },""")

        
def copy2colab(chapter,filename):
    cr=CellReader(filename)
    first=True
    with open(os.path.join('ColabNotebooks',f'Chapter{chapter}',os.path.split(filename)[1]),'w') as f:
        for line in cr.header:
            f.write(line)
        # add_image(f)
        for cell in cr:
            if cell is None:
                break
            elif len(cr.get_source()) and '(hidden on colab)' in cr.get_source()[0]:
                pass
            elif len(cr.get_source()) and 'SETUP SLEEPY' in cr.get_source()[0]:
                f.write('\n' if first else ',\n')
                write_colab_setup(f)
            else:
                f.write('\n' if first else ',\n')
                write_cell(f,cell,colab=True)
            first=False
        f.write('\n')
        for line in cr.footer:
            f.write(line)
    cr.__exit__()
    
def copy2JupyterBook(chapter,filename):
    cr=CellReader(filename)
    first=True
    with open(os.path.join('JupyterBook',f'Chapter{chapter}',os.path.split(filename)[1]),'w') as f:
        for line in cr.header:
            f.write(line)
        for k,cell in enumerate(cr):
            if cell is None:
                break
            elif len(cr.get_source()) and '(hidden on webpage)' in cr.get_source()[0]:
                pass
            elif len(cr.get_source()) and 'SETUP SLEEPY' in cr.get_source()[0]:
                f.write('\n' if first else ',\n')
                write_book_setup(f)
            else:
                f.write('\n' if first else ',\n')
                write_cell(f,cell)
            if first:
                first=False
                f.write(',\n')
                add_links(f,filename)
                
        f.write('\n')
        for line in cr.footer:
            f.write(line)
    cr.__exit__()
    
  

if __name__=='__main__':
    from SetupNotebooks import *
    
    directory='/Users/albertsmith/Documents/GitHub/SLEEPY_tutorial'
    if not(os.path.exists(directory)):
        directory=directory.replace('GitHub','GitHub.nosync')
    
    #Load the file_records (modified times) if it exists
    file_record={}
    if os.path.exists('file_record.txt'):
        with open('file_record.txt','r') as f:
            for line in f:
                key,value=line.strip().split('\t')
                file_record[key]=int(value)
            
    with open('file_record.txt','w') as f:
        for chapter in range(1,7):
            print(f'Chapter: {chapter}')
            for filename in os.listdir(os.path.join(directory,'Chapters',f'Chapter{chapter}')):
                filename=os.path.join(directory,'Chapters',f'Chapter{chapter}',filename)
                if '.ipynb' in filename and filename[-6:]=='.ipynb':
                    mt=int(os.path.getmtime(filename))
                    if not(filename in file_record and mt==file_record[os.path.split(filename)[1]]):
                        #new or modified notebooks get copied
                        copy2colab(chapter,filename)
                        copy2JupyterBook(chapter,filename)
                    
                    f.write(f'{filename}\t{mt}\n')
        
        