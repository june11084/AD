/****** Script for SelectTopNRows command from SSMS  ******/
SELECT f.[name]
	  ,n.[name]
	  ,a.[node_id]
      ,r.[name]
      ,a.[event]
      ,a.[timestamp]
  FROM [MESHIFY_LANDING_RD].[meshify].[alarms] a
  JOIN [MESHIFY_LANDING_RD].[meshify].[nodes] n on n.id = a.node_id
  JOIN [MESHIFY_LANDING_RD].[meshify].[folders] f on n.folder_id = f.id
  JOIN [MESHIFY_LANDING_RD].[meshify].[rules] r on r.id = a.rule_id