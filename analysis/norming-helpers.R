load_data = function(versions, type) {
    versions %>% 
    map(~ read_csv(glue('../data/{.x}/{type}.csv'), col_types = cols())) %>% 
    bind_rows
}

load_genie = function(versions) {
    full_pdf = load_data(versions, 'participants') %>% 
        filter(completed) %>% 
        # filter(wid != "w963f6eb")  %>%  # not actually completed, not sure what's going on
        select(-c(consideration_time, reflection_time, n_considered, n_unconsidered, version, bonus, completed)) %>%
        mutate(
            obligatory_check = factor(if_else(comp_radio, 'incorrect', 'correct', 'empty'),
                                      levels=c('empty', 'incorrect', 'correct')),
            # ignored_comprehension = obligatory_check == "empty" & is.na(comp_free)
        )

    full_df = load_data(versions, 'outcomes') %>% rename(scenario = prompt_id)
    full_scenarios = load_data(versions, 'scenarios') %>% 
        mutate(
            considered = map(considered, fromJSON),
            n_considered = lengths(considered)
        ) %>% rename(scenario = prompt_id)


    full_slider = load_data(versions, 'slider_check')

    # table(full_pdf$obligatory_check)
    pdf = full_pdf %>% 
        filter(obligatory_check == "correct") %>% 
        mutate(
            subj = row_number(),
            control = factor(control, levels=c("low", "high"))
        )

    pdf2 = select(pdf, wid, subj, control)

    unk_df = full_df %>% 
        right_join(pdf2) %>%  # also drops excluded participants
        filter(scenario != "PRACTICE") %>%
        group_by(wid) %>% 
        mutate(
            scenario = tolower(scenario),
            evaluation_z = zscore(evaluation),
            abs_eval = abs(evaluation),
            abs_eval_z = zscore(abs_eval),
            consideration=if_else(considered, "considered", "unconsidered")
        ) %>% 
        ungroup()

    df = filter(unk_df, outcome != "UNK")
    n_drop = nrow(full_pdf) - nrow(pdf)
    n_drop_unk = nrow(unk_df) - nrow(df)
    n_trial = nrow(df)

    scenarios = full_scenarios %>%
        right_join(pdf2)  %>% 
        filter(scenario != "PRACTICE") %>%
        group_by(wid) %>% 
        mutate(
            scenario = tolower(scenario),
            scenario_evaluation_z = zscore(scenario_evaluation),
            trial_number = row_number(),
        ) %>% 
        ungroup()

    left_join(df, select(scenarios, wid, scenario, scenario_evaluation, scenario_evaluation_z))
}

load_bestworst = function(versions) {
    full_pdf = load_data(versions, 'participants') %>% 
        filter(completed) %>% 
        select(-c(consideration_time, reflection_time, n_considered, n_unconsidered, version, bonus, completed))

    full_df = load_data(versions, 'outcomes') %>% rename(scenario = prompt_id)
    full_scenarios = load_data(versions, 'scenario') %>% rename(scenario = prompt_id)


    full_slider = load_data(versions, 'slider_check')

    # table(full_pdf$obligatory_check)
    pdf = full_pdf %>% 
        mutate(
            subj = row_number(),
            target = factor(target, levels=c("worst", "best"))
        )

    pdf2 = select(pdf, wid, subj, target)

    unk_df = full_df %>% 
        right_join(pdf2) %>%  # also drops excluded participants
        filter(scenario != "PRACTICE") %>%
        group_by(wid) %>% 
        mutate(
            scenario = tolower(scenario),
            evaluation_z = zscore(evaluation),
            abs_eval = abs(evaluation),
            abs_eval_z = zscore(abs_eval),
            consideration=if_else(considered, "considered", "unconsidered")
        ) %>% 
        ungroup()

    df = filter(unk_df, outcome != "UNK")
    n_drop = nrow(full_pdf) - nrow(pdf)
    n_drop_unk = nrow(unk_df) - nrow(df)
    n_trial = nrow(df)

    scenarios = full_scenarios %>%
        right_join(pdf2)  %>% 
        filter(scenario != "PRACTICE") %>%
        group_by(wid) %>% 
        mutate(
            scenario = tolower(scenario),
            final_outcome = outcome,
            # scenario_evaluation_z = zscore(scenario_evaluation),
            trial_number = row_number(),
        ) %>% 
        ungroup()

    left_join(df, select(scenarios, wid, scenario, final_outcome))
}