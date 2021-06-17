import cudf

# Global dataframe which will have each incoming flowfiles contents appended to it
global_df = cudf.DataFrame() # This is for demo purposes. Real application would use a controller service here

def describe(processor):
    processor.setDescription("Loads CSV Data into cuDF")

def onInitialize(processor):
    processor.setSupportsDynamicProperties()

# Reads the contents of the incoming flowfile
class ContentExtract(object):
    def __init__(self):
        self.content = None

    def process(self, input_stream):
        self.content = input_stream.read()
        return len(self.content)


def onTrigger(context, session):

    global global_df

    # In this situation a flowfile has already been created and transferred
    # here therefore we want to "get" it which means read it contents to 
    # be processed here
    flow_file = session.get()

    csv_flowfile = ContentExtract()

    # 'flow_file' is the pointer the to bytes. 'get' doesn't perform the actual read it is deferred until 'process' is called
    session.read(flow_file, csv_flowfile)

    # From this point its just like any other way to get messages into cuDF
    gdf = cudf.read_csv(path_or_buf=csv_flowfile.content,
                         lines=True,
                         engine="cudf")
    
    # Append the dataframe to the global dataframe
    global_df = global_df.append(gdf)

    # Attributes are 'headers' placed on top of a flowfile. They are not required just showing them here
    # the interesting thing about them is they can be used as a liteweight mechanism for routing flowfiles
    # downstream without having to 'bust open' the entire contents of the flowfile if it is quite large.
    # Think of them like an index for the content of a flowfile
    flow_file.addAttribute("num_rows", str(gdf.shape[0]))

    print("Individual # Rows: " + str(gdf.shape[0]))
    print("Global DataFrame: # Rows: " + str(global_df.shape[0]))

    # Notice we have no 'write content' here. This is to show options where the content of a flowfile
    # does not require being altered. Here we used it already and have no reason to alter it further
    session.transfer(flow_file, REL_SUCCESS)
