

def create_graph_classic(infection_events_df):
    # Ensure PIDs are converted to float IDs using the tick value
    if not infection_events_df.empty:
        #as int
        infection_events_df['pid']=infection_events_df['pid'].astype(str)
        infection_events_df['contact_pid']=infection_events_df['contact_pid'].astype(str)
        #add alias column for pid as f"{pid}.{tick}" cast the resulting string to float
        infection_events_df['alias_pid'] = (infection_events_df['pid'] + '.' + infection_events_df['tick'].astype(str))
    #get dictinoary from pid and alias columns
    #alias_dict = infection_events_df.set_index('pid')['alias'].to_dict()
    #alias_dict = dict(zip(infection_events_df['pid'], infection_events_df['alias']))
    # Initialize an empty dictionary to keep track of the current alias for each target string (pid)
    alias_map = {}

    # Prepare a list to collect the alias_contact values for each row.
    alias_contact = []

    # Iterate over the DataFrame rows in order.
    # Use itertuples for faster iteration.
    for row in infection_events_df.itertuples(index=False):
        # Look up the current alias for the contact_pid.
        # If the target hasn't been seen yet, you can return np.nan or some default value.
        current_alias = alias_map.get(row.contact_pid, "-1")
        alias_contact.append(current_alias)

        # Now update the mapping for this row's pid using the alias_pid value.
        # Because the first occurrence for any pid is always in the pid column, this sets or updates the alias.
        alias_map[row.pid] = row.alias_pid

    # Assign the list as a new column in the DataFrame.
    infection_events_df['alias_contact'] = alias_contact

    # --- 1. Analyze Full Graph ---
    G_full = nx.DiGraph()
    #G_full = nx.MultiDiGraph()
    if not infection_events_df.empty:
        #only rows with contact_pid != -1
        # Filter the DataFrame to include only rows where contact_pid is not -1
        no_imports=infection_events_df[infection_events_df['contact_pid'] != "-1"]
        # OPTIMIZATION: Use add_edges_from for much faster graph construction
        edges = zip(no_imports['alias_contact'], no_imports['alias_pid'], no_imports['tick'])
        G_full.add_edges_from((u, v, {'tick': tick}) for u, v, tick in edges)

        G_full.add_edges_from(edges)
    return G_full 


def create_component_table(infection_df):
    G_full=create_graph_classic(infection_df)
    full_cascade_components = list(nx.weakly_connected_components(G_full))
    #create table with pid, tick, component_id
    component_data = []
    for i, component in enumerate(full_cascade_components):
        for node in component:
            component_data.append({'alias_pid': node, 'component_id': i})
    component_df = pd.DataFrame(component_data)
    return component_df



#process the epihiper components 
#variant_labels=[v1,v2,v3,v4,v5,v6,v7,v8,v9,v10]
def create_variant_labels(epihiper_df, variant_labels):
    #filter epihiper_df to exit_state that starts with 'E'
    infection_df = epihiper_df[epihiper_df['exit_state'].str.startswith('E')]
    component_df=create_component_table(infection_df)
    #merge with infection_df on alias_pid
    merged_df = pd.merge(infection_df, component_df, on='alias_pid', how='left')
    #now I need to apply the component_id to all events that apply to the pid until the next infection event for that pid
    #sort by pid and tick
    merged_df = merged_df.sort_values(by=['pid', 'tick'])
    #forward fill component_id for each pid
    merged_df['component_id'] = merged_df.groupby('pid')['component_id'].ffill
    #sort by tick
    merged_df = merged_df.sort_values(by='tick')
    #create variant label as "variant_{component_id}_{first_tick_in_component}"
    first_ticks = merged_df.groupby('component_id')['tick'].min().reset_index()
    first_ticks = first_ticks.rename(columns={'tick': 'first_tick'})
    merged_df = pd.merge(merged_df, first_ticks, on='component_id', how='left')
    merged_df['variant_label'] = 'variant_' + merged_df['component_id'].astype(str) + '_' + merged_df['first_tick'].astype(str)
    return merged_df