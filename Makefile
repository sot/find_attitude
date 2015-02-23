# Define the task name
TASK = find_attitude

# Set Flight environment to be SKA.  The other choice is TST.  Include the
# Makefile.FLIGHT make file that does most of the hard work
FLIGHT = SKA
include /proj/sot/ska/include/Makefile.FLIGHT

WWW  = $(INSTALL)/www
# Define the installed executables for the task.  This directory is reserved
# for documented tools and not for dedicated scripts, one-off codes etc
DOC = doc/_build/html/*

.PHONY: doc

install_doc:
	mkdir -p $(INSTALL_DOC)
	rsync --archive --times $(DOC)   $(INSTALL_DOC)/

install_data:
	mkdir -p $(INSTALL_DATA)
	rsync --size-only /export/tom/data/find_attitude/distances.h5 $(INSTALL_DATA)/
	ln -s -f /export/tom/data/find_attitude/distances.h5 $(INSTALL_DATA)/distances-kadi-local.h5

