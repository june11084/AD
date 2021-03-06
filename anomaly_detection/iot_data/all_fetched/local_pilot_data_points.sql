/****** Script for SelectTopNRows command from SSMS  ******/
SELECT f.[name]
	  ,f.[code]
	  ,f.[locality]
	  ,f.[region]
	  ,n.[name]
      ,d.[node_id]
      ,d.[channel]
      ,d.[timestamp]
      ,d.[value]
  FROM [MESHIFY_LANDING_RD].[meshify].[data_points] d
  JOIN [MESHIFY_LANDING_RD].[meshify].[nodes] n on n.id = d.node_id
  JOIN [MESHIFY_LANDING_RD].[meshify].[folders] f on f.id = n.folder_id